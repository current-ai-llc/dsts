import { generateText } from 'ai';
import {
  GEPAOptions,
  GEPAResult,
  GEPAAdapter,
  Candidate,
  Logger,
  LanguageModel,
  LanguageModelConfig,
  ComponentSelector,
  BatchSampler,
  CandidateSelector,
} from './types';
import {
  buildParetoFront,
  hypervolume2D,
  average,
  avgVec,
  selectProgramCandidateFromInstanceFronts,
} from './pareto-utils';
import {
  RoundRobinComponentSelector,
  AllComponentSelector,
  EpochShuffledBatchSampler,
  CurrentBestCandidateSelector,
  ParetoCandidateSelector,
} from './strategies';
import { DefaultAdapter } from './adapters/default-adapter';
import { FilePersistence, CheckpointState } from './persistence';

/**
 * GEPA Optimizer - Multi-objective optimization through reflective evolution
 */
export class GEPA {
  private adapter!: GEPAAdapter;
  private reflectionLM!: LanguageModel;
  private componentSelector: ComponentSelector;
  private batchSampler: BatchSampler;
  private candidateSelector: CandidateSelector;
  private logger?: Logger;
  private perfectScore: number;
  private skipPerfectScore: boolean;
  private maxMetricCalls?: number;
  private maxIterations?: number;
  private maxBudgetUSD?: number;
  private verbose: boolean = false;
  private latestReflectionPrompt: string | null = null;
  private latestReflectionSummary: string | null = null;
  private reflectionModelName?: string;
  private reflectionLMProviderOptions?: Record<string, any>;
  private taskModelName?: string;
  private tieEpsilon: number = 0;
  private rng: () => number;
  private rngState: number = 0;
  private persistence?: FilePersistence;
  private checkpointEveryIterations: number = 1;
  
  // Stats
  private totalMetricCalls: number = 0;
  private totalCostUSD: number = 0;
  private history: Array<{
    iteration: number;
    candidate: Candidate;
    scores: Record<string, number>;
    accepted: boolean;
  }> = [];

  constructor(private options: GEPAOptions) {
    // Initialize adapter
    if (options.adapter) {
      this.adapter = options.adapter;
    } 
    
    if (options.taskLM) {
      const taskLMConfig = options.taskLM;
      if (typeof taskLMConfig === 'string') {
        if (!this.adapter) {
          this.adapter = new DefaultAdapter({ model: taskLMConfig });
        }
        this.taskModelName = taskLMConfig;
      } else if (typeof taskLMConfig === 'function') {
        // Cannot introspect model name from function; adapter must handle task calls
      } else if (typeof taskLMConfig === 'object' && 'model' in taskLMConfig) {
        if (!this.adapter) {
          const { model, apiKey, temperature, costEstimator, maxConcurrency, ...passthrough } = taskLMConfig
          this.adapter = new DefaultAdapter({
            model,
            apiKey,
            temperature,
            costEstimator,
            maxConcurrency,
            ...passthrough,
          });
        }
        this.taskModelName = taskLMConfig.model;
      } else {
        throw new Error('taskLM must be a string, function, or an object with a model property');
      }
    }

    if (!this.adapter) {
      throw new Error('Either adapter or taskLM must be provided');
    }

    // Initialize reflection LM
    this.reflectionLM = this.initializeLanguageModel(options.reflectionLM);

    // Initialize strategies
    const seed = options.seed ?? 0;
    this.rng = this.createRNG(seed);
    this.rngState = seed || 123456789;

    // Persistence
    if (options.persistence?.dir) {
      this.persistence = new FilePersistence(
        options.persistence.dir,
        options.persistence.archiveFile
      );
      this.checkpointEveryIterations = options.persistence.checkpointEveryIterations ?? 1;
    }

    if (typeof options.componentSelector === 'string') {
      this.componentSelector = options.componentSelector === 'all' 
        ? new AllComponentSelector()
        : new RoundRobinComponentSelector();
    } else {
      this.componentSelector = options.componentSelector ?? new RoundRobinComponentSelector();
    }

    this.batchSampler = new EpochShuffledBatchSampler(
      options.reflectionMinibatchSize ?? 3,
      this.rng
    );

    if (options.candidateSelectionStrategy === 'pareto') {
      this.candidateSelector = new ParetoCandidateSelector(this.rng);
    } else {
      this.candidateSelector = new CurrentBestCandidateSelector();
    }

    this.logger = options.logger;
    this.perfectScore = options.perfectScore ?? 1;
    this.skipPerfectScore = options.skipPerfectScore ?? true;
    this.maxMetricCalls = options.maxMetricCalls;
    this.maxIterations = options.maxIterations;
    this.maxBudgetUSD = options.maxBudgetUSD;
    this.verbose = options.verbose ?? false;
    this.reflectionLMProviderOptions = options.reflectionLMProviderOptions;

    // Ensure we have at least one stopping criterion
    if (this.maxMetricCalls === undefined && this.maxIterations === undefined && this.maxBudgetUSD === undefined) {
      throw new Error('You must set at least one of maxMetricCalls, maxIterations, or maxBudgetUSD');
    }
  }

  /**
   * Run GEPA optimization
   */
  async optimize(): Promise<GEPAResult> {
    const startTime = Date.now();
    const { seedCandidate, trainset, valset, persistence } = this.options;
    const validationSet = valset ?? trainset;

    // Resume from checkpoint if requested
    let iteration = 0;
    const candidates: Array<{
      candidate: Candidate;
      scores: Record<string, number>;
      scalarScore: number;
      parent?: number;
    }> = [];
    const perInstanceScores: number[][] = [];

    if (persistence?.resume && this.persistence) {
      const cp = this.persistence.loadCheckpoint();
      if (cp) {
        iteration = cp.iteration;
        this.totalMetricCalls = cp.totalMetricCalls;
        this.totalCostUSD = cp.totalCostUSD;
        this.rngState = cp.rngState;
        candidates.push(...cp.candidates);
        perInstanceScores.push(...cp.perInstanceScores);
      }
    }

    this.log('info', 'Starting GEPA optimization', {
      trainSize: trainset.length,
      valSize: validationSet.length,
      components: Object.keys(seedCandidate),
      maxMetricCalls: this.maxMetricCalls,
      maxBudgetUSD: this.maxBudgetUSD,
    });

    // Initialize candidates and scores (if not resumed)
    if (candidates.length === 0) {
    const seedEval = await this.evaluateCandidate(seedCandidate, validationSet);
    candidates.push({
      candidate: seedCandidate,
      scores: seedEval.scores,
        scalarScore: seedEval.scalarScore,
         });
      perInstanceScores.push(seedEval.instanceScores);
    }

    if (this.candidateSelector instanceof ParetoCandidateSelector) {
    this.candidateSelector.setInstanceScores(perInstanceScores);
    }

       let stagnation = 0;
    const earlyStoppingTrials = 5;

    // Main optimization loop
    while (true) {
      // Metric calls cap
      if (this.maxMetricCalls !== undefined && this.totalMetricCalls >= this.maxMetricCalls) {
        this.log('info', 'Max metric calls reached', { totalMetricCalls: this.totalMetricCalls, maxMetricCalls: this.maxMetricCalls });
        break;
      }

      iteration++;

      // Budget cap check
      if (this.maxBudgetUSD !== undefined && this.totalCostUSD >= this.maxBudgetUSD) {
        this.log('info', 'Max budget reached', { totalCostUSD: this.totalCostUSD, maxBudgetUSD: this.maxBudgetUSD });
        break;
      }

      // Iteration cap
      if (this.maxIterations !== undefined && iteration > this.maxIterations) {
        this.log('info', 'Max iterations reached', { iteration, maxIterations: this.maxIterations });
        break;
      }

      // Select parent candidate
      const paretoIndices = buildParetoFront(
        candidates.map((c, idx) => ({ idx, scores: c.scores })),
        this.tieEpsilon
      ).map(p => p.idx);

      const parentIdx = this.candidateSelector.selectCandidate(
        candidates,
        paretoIndices
      );
      const parent = candidates[parentIdx];

      // Get minibatch for reflection
      const minibatch = this.batchSampler.nextBatch(trainset, iteration);

      // Skip if perfect score on minibatch
      if (this.skipPerfectScore) {
        const minibatchEval = await this.adapter.evaluate(
          minibatch,
          parent.candidate,
          false
        );
        // accumulate cost
        if (minibatchEval.metrics) {
          const c = minibatchEval.metrics.reduce((s, m) => s + (m.cost_usd ?? 0), 0);
          this.totalCostUSD += c;
        }
        const avgScore = average(minibatchEval.scores);
        if (avgScore >= this.perfectScore) {
          this.log('debug', 'Skipping iteration due to perfect score', { iteration, avgScore });
          continue;
        }
      }

      // Generate new candidate through reflection
      const componentsToUpdate = this.componentSelector.selectComponents(
        parent.candidate,
        iteration
      );

      const newCandidate = await this.reflectAndPropose(
        parent.candidate,
        minibatch,
        componentsToUpdate
      );

      // Evaluate new candidate on minibatch for acceptance
      const parentMinibatchEval = await this.adapter.evaluate(
        minibatch,
        parent.candidate,
        false
      );
      if (parentMinibatchEval.metrics) {
        const c1 = parentMinibatchEval.metrics.reduce((s, m) => s + (m.cost_usd ?? 0), 0);
        this.totalCostUSD += c1;
      }
      const childMinibatchEval = await this.adapter.evaluate(
        minibatch,
        newCandidate,
        false
      );
      if (childMinibatchEval.metrics) {
        const c2 = childMinibatchEval.metrics.reduce((s, m) => s + (m.cost_usd ?? 0), 0);
        this.totalCostUSD += c2;
      }

      const parentMinibatchSum = parentMinibatchEval.scores.reduce((a, b) => a + b, 0);
      const childMinibatchSum = childMinibatchEval.scores.reduce((a, b) => a + b, 0);

      // Accept if child is better
      const accepted = childMinibatchSum > parentMinibatchSum + this.tieEpsilon;

      if (accepted) {
        // Full evaluation on validation set
        const childEval = await this.evaluateCandidate(newCandidate, validationSet);
        
        candidates.push({
          candidate: newCandidate,
          scores: childEval.scores,
          scalarScore: childEval.scalarScore,
          parent: parentIdx,
        });

        perInstanceScores.push(childEval.instanceScores);
        if (this.candidateSelector instanceof ParetoCandidateSelector) {
          this.candidateSelector.setInstanceScores(perInstanceScores);
        }

        stagnation = 0;
        
        this.log('info', 'Accepted new candidate', {
          iteration,
          improvement: childMinibatchSum - parentMinibatchSum,
          valScore: childEval.scalarScore,
        });
      } else {
        stagnation++;
        this.log('debug', 'Rejected candidate', {
          iteration,
          parentScore: parentMinibatchSum,
          childScore: childMinibatchSum,
        });
      }

      // Record history
      this.history.push({
        iteration,
        candidate: newCandidate,
        scores: accepted ? candidates[candidates.length - 1].scores : {},
        accepted,
      });

        // Log Pareto and hypervolume each iteration (2D only)
      const iterPareto = buildParetoFront(
        candidates.map((c, idx) => ({ idx, scores: c.scores })),
        this.tieEpsilon
      );
      const iterHv = hypervolume2D(iterPareto.map(p => candidates[p.idx].scores));
      if (iterHv !== null) {
        this.log('info', 'Iter hypervolume (2D)', { iteration, hypervolume2D: iterHv });
      }

      // Verbose printing
      if (this.verbose) {
        const paretoLines = iterPareto.map(p => {
          const c = candidates[p.idx];
          const corr = c.scores.correctness?.toFixed(4);
          const lat = c.scores.latency?.toFixed(2);
          const sc = c.scalarScore.toFixed(4);
          return `#${p.idx}: correctness=${corr}, latency=${lat}, scalar=${sc}`;
        }).join('\n');
        const header = `\n=== Iteration ${iteration} Summary ===`;
        const hvLine = `Hypervolume(2D): ${iterHv ?? 'n/a'}`;
        const callsCost = `Calls: ${this.totalMetricCalls}  |  Cost USD: ${this.totalCostUSD.toFixed(4)}`;
        const promptBlock = this.latestReflectionPrompt ? `\n--- Latest Reflection Prompt ---\n${this.latestReflectionPrompt}` : '';
        const summaryBlock = this.latestReflectionSummary ? `\n--- Reflection Summary ---\n${this.latestReflectionSummary}` : '';
        console.log(`${header}\n${hvLine}\n${callsCost}\n--- Pareto Front ---\n${paretoLines}${promptBlock}${summaryBlock}\n`);
      }

      // Archive + checkpoint
      if (this.persistence) {
        this.persistence.appendArchive({
          ts: Date.now(),
          iteration,
          event: accepted ? 'accepted' : 'rejected',
          data: accepted ? candidates[candidates.length - 1] : undefined,
        });
        if (iteration % this.checkpointEveryIterations === 0) {
          const cp: CheckpointState = {
            iteration,
            totalMetricCalls: this.totalMetricCalls,
            totalCostUSD: this.totalCostUSD,
            rngState: this.rngState,
            candidates,
            perInstanceScores,
          };
          this.persistence.saveCheckpoint(cp);
        }
      }

      // Early stopping
      if (stagnation >= earlyStoppingTrials) {
        this.log('info', 'Early stopping due to stagnation', { stagnation });
        break;
      }
    }

    // Build final Pareto front
    const paretoFront = buildParetoFront(
      candidates.map((c, idx) => ({ idx, scores: c.scores })),
      this.tieEpsilon
    );

    // Compute hypervolume for 2D objectives if available (correctness, speed)
    const hv = hypervolume2D(
      paretoFront.map(p => candidates[p.idx].scores)
    );
    if (hv !== null) {
      this.log('info', 'Pareto hypervolume (2D)', { hypervolume2D: hv });
    }

    if (this.persistence) {
      this.persistence.appendArchive({ ts: Date.now(), iteration, event: 'finish', data: { hv } });
      const cp: CheckpointState = {
        iteration,
        totalMetricCalls: this.totalMetricCalls,
        totalCostUSD: this.totalCostUSD,
        rngState: this.rngState,
        candidates,
        perInstanceScores,
      };
      this.persistence.saveCheckpoint(cp);
    }

    // Find best candidate
    let bestIdx = 0;
    let bestScore = candidates[0].scalarScore;
    for (let i = 1; i < candidates.length; i++) {
      if (candidates[i].scalarScore > bestScore) {
        bestScore = candidates[i].scalarScore;
        bestIdx = i;
      }
    }

    const result: GEPAResult = {
      bestCandidate: candidates[bestIdx].candidate,
      bestScore,
      paretoFront: paretoFront.map(p => candidates[p.idx]),
      iterations: iteration,
      totalMetricCalls: this.totalMetricCalls,
      history: this.history,
      totalCostUSD: this.totalCostUSD,
    };

    const duration = Date.now() - startTime;
    this.log('info', 'GEPA optimization complete', {
      duration,
      iterations: iteration,
      totalMetricCalls: this.totalMetricCalls,
      totalCostUSD: this.totalCostUSD,
      bestScore,
      paretoSize: paretoFront.length,
    });

    return result;
  }

  /**
   * Evaluate candidate on full dataset
   */
  private async evaluateCandidate(
    candidate: Candidate,
    dataset: any[]
  ): Promise<{
    scores: Record<string, number>;
    scalarScore: number;
    instanceScores: number[];
  }> {
    const evalBatch = await this.adapter.evaluate(dataset, candidate, false);
    this.totalMetricCalls += dataset.length;
    // Accumulate cost if available
    if (evalBatch.metrics) {
      const cost = evalBatch.metrics.reduce((sum, m) => sum + (m.cost_usd ?? 0), 0);
      this.totalCostUSD += cost;
    }

    // Primary objective: correctness (average score)
    const correctness = average(evalBatch.scores);

    // Secondary objective: latency (we maximize by storing negative avg latency)
    let avgLatency = 0;
    if (evalBatch.metrics && evalBatch.metrics.length > 0) {
      const latencies = evalBatch.metrics.map(m => m.latency_ms ?? 0);
      avgLatency = average(latencies);
    }

    // Always include both metrics for consistent Pareto comparisons
    const scores: Record<string, number> = { correctness, latency: -Math.max(0, avgLatency) };

    const scalarScore = correctness; // keep scalar selection on correctness by default

    return {
      scores,
      scalarScore,
      instanceScores: evalBatch.scores,
    };
  }

  /**
   * Reflect on performance and propose new candidate
   */
  private async reflectAndPropose(
    candidate: Candidate,
    minibatch: any[],
    componentsToUpdate: string[]
  ): Promise<Candidate> {
    // Evaluate with traces for reflection
    const evalBatch = await this.adapter.evaluate(minibatch, candidate, true);
    this.totalMetricCalls += minibatch.length;
    if (evalBatch.metrics) {
      const cost = evalBatch.metrics.reduce((sum, m) => sum + (m.cost_usd ?? 0), 0);
      this.totalCostUSD += cost;
    }

    // Check if adapter has custom proposal function
    if (this.adapter.proposeNewTexts) {
      const reflectiveDataset = this.adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        componentsToUpdate
      );
      // Try to generate a human-readable summary for verbose mode
      await this.generateAndStoreReflectionSummary(reflectiveDataset);
      return await this.adapter.proposeNewTexts(
        candidate,
        reflectiveDataset,
        componentsToUpdate
      );
    }

    // Default reflection using LLM
    const reflectiveDataset = this.adapter.makeReflectiveDataset(
      candidate,
      evalBatch,
      componentsToUpdate
    );

    const newCandidate = { ...candidate };

    // Build prompts per component and remember the latest/combined prompt
    const promptSections: string[] = [];
    for (const component of componentsToUpdate) {
      const examples = reflectiveDataset[component] || [];
      if (examples.length === 0) continue;

      const prompt = this.buildReflectionPrompt(
        component,
        candidate[component],
        examples
      );
      promptSections.push(`----- Component: ${component} -----\n${prompt}`);

      const newText = await this.reflectionLM(prompt);
      newCandidate[component] = newText.trim();
    }
    this.latestReflectionPrompt = promptSections.length > 0 ? promptSections.join("\n\n") : null;

    // Generate user-readable summary for this iteration (verbose mode)
    await this.generateAndStoreReflectionSummary(reflectiveDataset);

    return newCandidate;
  }

  /**
   * Build prompt for reflection
   */
  private buildReflectionPrompt(
    componentName: string,
    currentText: string,
    examples: Array<Record<string, any>>
  ): string {
    const hint = this.options.reflectionHint
      ? `\nAdditional guidance:\n${this.options.reflectionHint}\n`
      : ''
    return `You are tasked with improving a text component based on execution feedback.${hint}

Component Name: ${componentName}
Current Text:
${currentText}

Execution Examples and Feedback:
${JSON.stringify(examples, null, 2)}

Based on the feedback above, propose an improved version of the text that addresses the issues identified.
Focus on:
1. Fixing errors mentioned in the feedback
2. Improving clarity and specificity
3. Better handling edge cases
4. Maintaining the original intent while improving execution

Provide only the improved text, without any explanation or markdown formatting:`;
  }

  /**
   * Generate and store a human-readable summary of reflection feedback
   */
  private async generateAndStoreReflectionSummary(reflectiveDataset: Record<string, Array<Record<string, any>>>): Promise<void> {
    if (!this.verbose) {
      this.latestReflectionSummary = null;
      return;
    }
    try {
      const summaryPrompt = `You will receive reflection feedback grouped by component from an optimization iteration. Write a concise, user-readable summary that explains:
- The main issues observed in the feedback
- The specific changes the next candidate will try for each component
- Any trade-offs or uncertainties to watch for next iteration
Use short paragraphs and bullet points. Avoid code fences. Feedback JSON follows:\n\n${JSON.stringify(reflectiveDataset, null, 2)}`;

      if (!this.taskModelName) {
        this.latestReflectionSummary = 'Task LM not configured; skipped summary.';
        return;
      }

      const result = await generateText({
        model: this.taskModelName,
        messages: [{ role: 'user', content: summaryPrompt }],
        temperature: 0.3,
      });
      this.latestReflectionSummary = (result.text || '').trim();
    } catch (e) {
      this.latestReflectionSummary = 'Summary generation failed.';
    }
  }

  /**
   * Initialize language model from config
   */
  private initializeLanguageModel(config?: LanguageModelConfig): LanguageModel {
    if (!config) {
      throw new Error('reflectionLM must be provided');
    }

    if (typeof config === 'function') {
      return config;
    }

    const model = typeof config === 'string' ? config : config.model;
    const temperature = typeof config === 'object' ? config.temperature : 0.7;
    this.reflectionModelName = model;

    return async (prompt: string) => {
      const result = await generateText({
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature,
        providerOptions: this.reflectionLMProviderOptions,
      });
      return result.text;
    };
  }

  /**
   * Create seeded random number generator
   */
  private createRNG(seed: number): () => number {
    this.rngState = seed || 123456789;
    return () => {
      // xorshift
      this.rngState ^= this.rngState << 13;
      this.rngState ^= this.rngState >>> 17;
      this.rngState ^= this.rngState << 5;
      return ((this.rngState >>> 0) as number) / 4294967296;
    };
  }

  /**
   * Log message
   */
  private log(level: string, message: string, data?: any): void {
    if (this.logger) {
      this.logger.log(level, message, data);
    }
  }
}

/**
 * Convenience function to run GEPA optimization
 */
export async function optimize(options: GEPAOptions): Promise<GEPAResult> {
  const gepa = new GEPA(options);
  return await gepa.optimize();
}