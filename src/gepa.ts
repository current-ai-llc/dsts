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
          const { model, apiKey, temperature, costEstimator, maxConcurrency, reflectOnAllTexts, ...passthrough } = taskLMConfig
          this.adapter = new DefaultAdapter({
            model,
            apiKey,
            temperature,
            costEstimator,
            maxConcurrency,
            reflectOnAllTexts,
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

      this.log('debug', 'Pareto front built', {
        iteration,
        totalCandidates: candidates.length,
        paretoIndices,
        paretoSize: paretoIndices.length
      });

      const parentIdx = this.candidateSelector.selectCandidate(
        candidates,
        paretoIndices
      );
      const parent = candidates[parentIdx];

      this.log('debug', 'Parent candidate selected', {
        iteration,
        parentIdx,
        parentScalarScore: parent.scalarScore,
        parentScores: parent.scores,
        parentComponentsPreview: Object.keys(parent.candidate).reduce((acc, key) => {
          acc[key] = parent.candidate[key].substring(0, 100) + (parent.candidate[key].length > 100 ? '...' : '');
          return acc;
        }, {} as Record<string, string>)
      });

      // Get minibatch for reflection
      const minibatch = this.batchSampler.nextBatch(trainset, iteration);

      // Skip if perfect score on minibatch
      if (this.skipPerfectScore) {
        this.log('debug', 'Evaluating parent for perfect score check', {
          iteration,
          parentIdx,
          candidatePreview: Object.keys(parent.candidate).reduce((acc, key) => {
            acc[key] = parent.candidate[key].substring(0, 50) + '...';
            return acc;
          }, {} as Record<string, string>)
        });

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
        
        this.log('debug', 'Perfect score check result', {
          iteration,
          avgScore,
          perfectScore: this.perfectScore,
          willSkip: avgScore >= this.perfectScore
        });
        
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
      
      // Log diff between parent and proposed candidate
      const oldSystem = parent.candidate.system || '';
      const newSystem = newCandidate.system || '';
      const proposedDiff = this.computeStringDiff(oldSystem, newSystem);
      
      if (proposedDiff.changed) {
        console.log('\n\x1b[1m=== Proposed System Prompt Changes (Iteration ' + iteration + ') ===\x1b[0m');
        console.log(`\x1b[32mLines added: ${proposedDiff.added}\x1b[0m`);
        console.log(`\x1b[31mLines removed: ${proposedDiff.removed}\x1b[0m`);
        console.log(`\x1b[33mLines modified: ${proposedDiff.modified}\x1b[0m`);
        if (proposedDiff.changes.length > 0) {
          console.log('\nChanges:');
          proposedDiff.changes.forEach(change => {
            if (change.startsWith('+')) {
              console.log(`\x1b[32m${change}\x1b[0m`);
            } else if (change.startsWith('-')) {
              console.log(`\x1b[31m${change}\x1b[0m`);
            } else {
              console.log(change);
            }
          });
        }
        console.log('\x1b[1m=========================\x1b[0m\n');
      } else {
        console.log(`\n\x1b[90m[Iteration ${iteration}] No changes proposed to system prompt\x1b[0m\n`);
      }

      // Evaluate new candidate on minibatch for acceptance
      this.log('debug', 'Evaluating parent candidate on minibatch', {
        iteration,
        parentIdx,
        minibatchSize: minibatch.length,
        parentCandidatePreview: Object.keys(parent.candidate).reduce((acc, key) => {
          acc[key] = parent.candidate[key].substring(0, 50) + '...';
          return acc;
        }, {} as Record<string, string>)
      });

      const parentMinibatchEval = await this.adapter.evaluate(
        minibatch,
        parent.candidate,
        false
      );
      if (parentMinibatchEval.metrics) {
        const c1 = parentMinibatchEval.metrics.reduce((s, m) => s + (m.cost_usd ?? 0), 0);
        this.totalCostUSD += c1;
      }

      this.log('debug', 'Evaluating new candidate on minibatch', {
        iteration,
        minibatchSize: minibatch.length,
        newCandidatePreview: Object.keys(newCandidate).reduce((acc, key) => {
          acc[key] = newCandidate[key].substring(0, 50) + '...';
          return acc;
        }, {} as Record<string, string>)
      });

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

      this.log('debug', 'Minibatch evaluation scores', {
        iteration,
        parentSum: parentMinibatchSum,
        childSum: childMinibatchSum,
        improvement: childMinibatchSum - parentMinibatchSum,
        tieEpsilon: this.tieEpsilon
      });

      // Accept if child is better
      const accepted = childMinibatchSum > parentMinibatchSum + this.tieEpsilon;

      if (accepted) {
        // Full evaluation on validation set
        const childEval = await this.evaluateCandidate(newCandidate, validationSet);
        
        const newCandidateEntry = {
          candidate: newCandidate,
          scores: childEval.scores,
          scalarScore: childEval.scalarScore,
          parent: parentIdx,
        };
        
        this.log('debug', 'Adding new candidate to array', {
          iteration,
          candidateIndex: candidates.length,
          parentIndex: parentIdx,
          newScalarScore: childEval.scalarScore,
          newScores: childEval.scores,
          candidatesArraySizeBefore: candidates.length,
          newCandidateComponentsPreview: Object.keys(newCandidate).reduce((acc, key) => {
            acc[key] = newCandidate[key].substring(0, 100) + (newCandidate[key].length > 100 ? '...' : '');
            return acc;
          }, {} as Record<string, string>)
        });

        candidates.push(newCandidateEntry);

        this.log('debug', 'Candidate array updated', {
          iteration,
          candidatesArraySizeAfter: candidates.length,
          newCandidateIndexInArray: candidates.length - 1,
          allCandidateScores: candidates.map((c, idx) => ({
            idx,
            scalarScore: c.scalarScore,
            parentIdx: c.parent
          }))
        });

        perInstanceScores.push(childEval.instanceScores);
        if (this.candidateSelector instanceof ParetoCandidateSelector) {
          this.candidateSelector.setInstanceScores(perInstanceScores);
        }

        stagnation = 0;
        
        // Log diff between old and new system prompts
        const oldSystem = parent.candidate.system || '';
        const newSystem = newCandidate.system || '';
        const diff = this.computeStringDiff(oldSystem, newSystem);
        
        this.log('info', 'Accepted new candidate', {
          iteration,
          improvement: childMinibatchSum - parentMinibatchSum,
          valScore: childEval.scalarScore,
        });
        
        if (diff.changed) {
          console.log('\n=== System Prompt Changes ===');
          console.log(`Lines added: ${diff.added}`);
          console.log(`Lines removed: ${diff.removed}`);
          console.log(`Lines modified: ${diff.modified}`);
          if (diff.changes.length > 0) {
            console.log('\nChanges:');
            diff.changes.forEach(change => console.log(change));
          }
          console.log('=========================\n');
        }
      } else {
        stagnation++;
        this.log('debug', 'Rejected candidate', {
          iteration,
          parentScore: parentMinibatchSum,
          childScore: childMinibatchSum,
          rejectedCandidatePreview: Object.keys(newCandidate).reduce((acc, key) => {
            acc[key] = newCandidate[key].substring(0, 100) + (newCandidate[key].length > 100 ? '...' : '');
            return acc;
          }, {} as Record<string, string>)
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
    this.log('debug', 'Evaluating candidate on full dataset', {
      datasetSize: dataset.length,
      candidateComponentsPreview: Object.keys(candidate).reduce((acc, key) => {
        acc[key] = candidate[key].substring(0, 100) + (candidate[key].length > 100 ? '...' : '');
        return acc;
      }, {} as Record<string, string>)
    });

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

    this.log('debug', 'Candidate evaluation complete', {
      correctness,
      avgLatency,
      scalarScore,
      scores,
      instanceScoresRange: evalBatch.scores.length > 0 ? [Math.min(...evalBatch.scores), Math.max(...evalBatch.scores)] : []
    });

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
    this.log('debug', 'Starting candidate generation', {
      componentsToUpdate,
      minibatchSize: minibatch.length,
      inputCandidatePreview: Object.keys(candidate).reduce((acc, key) => {
        acc[key] = candidate[key].substring(0, 100) + (candidate[key].length > 100 ? '...' : '');
        return acc;
      }, {} as Record<string, string>)
    });

    // Evaluate with traces for reflection
    const evalBatch = await this.adapter.evaluate(minibatch, candidate, true);
    this.totalMetricCalls += minibatch.length;
    if (evalBatch.metrics) {
      const cost = evalBatch.metrics.reduce((sum, m) => sum + (m.cost_usd ?? 0), 0);
      this.totalCostUSD += cost;
    }

    this.log('debug', 'Evaluation complete for reflection', {
      avgScore: average(evalBatch.scores),
      scoresRange: [Math.min(...evalBatch.scores), Math.max(...evalBatch.scores)],
      hasTraces: evalBatch.trajectories && evalBatch.trajectories.length > 0
    });

    // Check if adapter has custom proposal function
    if (this.adapter.proposeNewTexts) {
      this.log('debug', 'Using adapter custom proposal function');
      const reflectiveDataset = this.adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        componentsToUpdate
      );
      this.log('debug', 'Reflective dataset created', {
        components: Object.keys(reflectiveDataset),
        exampleCounts: Object.keys(reflectiveDataset).reduce((acc, comp) => {
          acc[comp] = reflectiveDataset[comp]?.length ?? 0;
          return acc;
        }, {} as Record<string, number>)
      });
      // Try to generate a human-readable summary for verbose mode
      await this.generateAndStoreReflectionSummary(reflectiveDataset);
      const newCandidate = await this.adapter.proposeNewTexts(
        candidate,
        reflectiveDataset,
        componentsToUpdate
      );
      this.log('debug', 'New candidate generated via adapter', {
        candidateChanged: JSON.stringify(candidate) !== JSON.stringify(newCandidate),
        newCandidatePreview: Object.keys(newCandidate).reduce((acc, key) => {
          acc[key] = newCandidate[key].substring(0, 100) + (newCandidate[key].length > 100 ? '...' : '');
          return acc;
        }, {} as Record<string, string>)
      });
      return newCandidate;
    }

    // Default reflection using LLM
    this.log('debug', 'Using default LLM reflection');
    const reflectiveDataset = this.adapter.makeReflectiveDataset(
      candidate,
      evalBatch,
      componentsToUpdate
    );

    this.log('debug', 'Reflective dataset created', {
      components: Object.keys(reflectiveDataset),
      exampleCounts: Object.keys(reflectiveDataset).reduce((acc, comp) => {
        acc[comp] = reflectiveDataset[comp]?.length ?? 0;
        return acc;
      }, {} as Record<string, number>)
    });

    const newCandidate = { ...candidate };

    // Build prompts per component and remember the latest/combined prompt
    const promptSections: string[] = [];
    for (const component of componentsToUpdate) {
      const examples = reflectiveDataset[component] || [];
      if (examples.length === 0) {
        this.log('debug', 'Skipping component with no examples', { component });
        continue;
      }

      this.log('debug', 'Generating new text for component', {
        component,
        exampleCount: examples.length,
        currentTextLength: candidate[component]?.length ?? 0
      });

      const prompt = this.buildReflectionPrompt(
        component,
        candidate[component],
        examples
      );
      promptSections.push(`----- Component: ${component} -----\n${prompt}`);

      const newText = await this.reflectionLM(prompt);
      const trimmedNewText = newText.trim();
      
      this.log('debug', 'Component text updated', {
        component,
        oldTextLength: candidate[component]?.length ?? 0,
        newTextLength: trimmedNewText.length,
        textChanged: candidate[component] !== trimmedNewText,
        oldTextPreview: candidate[component]?.substring(0, 100) + (candidate[component]?.length > 100 ? '...' : ''),
        newTextPreview: trimmedNewText.substring(0, 100) + (trimmedNewText.length > 100 ? '...' : '')
      });
      
      newCandidate[component] = trimmedNewText;
    }
    this.latestReflectionPrompt = promptSections.length > 0 ? promptSections.join("\n\n") : null;

    // Generate user-readable summary for this iteration (verbose mode)
    await this.generateAndStoreReflectionSummary(reflectiveDataset);

    const candidateChanged = JSON.stringify(candidate) !== JSON.stringify(newCandidate);
    this.log('debug', 'Candidate generation complete', {
      candidateChanged,
      componentsUpdated: componentsToUpdate.filter(comp => candidate[comp] !== newCandidate[comp]),
      newCandidatePreview: Object.keys(newCandidate).reduce((acc, key) => {
        acc[key] = newCandidate[key].substring(0, 100) + (newCandidate[key].length > 100 ? '...' : '');
        return acc;
      }, {} as Record<string, string>)
    });

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
    const {
      temperature = 0.7,
      tools,
      providerOptions,
      stopWhen,
      toolChoice,
      maxToolRoundtrips,
      experimentalTelemetry,
      ...otherOpts
    } = typeof config === 'object' ? config : {};
    this.reflectionModelName = model;

    return async (prompt: string) => {
      const generateTextOpts: any = {
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature,
        providerOptions: providerOptions || this.reflectionLMProviderOptions,
        ...otherOpts,
      };
      
      // Add optional fields if they exist
      if (tools) generateTextOpts.tools = tools;
      if (toolChoice) generateTextOpts.toolChoice = toolChoice;
      if (maxToolRoundtrips) generateTextOpts.maxToolRoundtrips = maxToolRoundtrips;
      if (stopWhen) generateTextOpts.stopWhen = stopWhen;
      if (experimentalTelemetry) generateTextOpts.experimental_telemetry = experimentalTelemetry;
      
      const result = await generateText(generateTextOpts);
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

  /**
   * Compute simple line-based diff between two strings
   */
  private computeStringDiff(oldStr: string, newStr: string): {
    changed: boolean;
    added: number;
    removed: number;
    modified: number;
    changes: string[];
  } {
    const oldLines = oldStr.split('\n');
    const newLines = newStr.split('\n');
    
    if (oldStr === newStr) {
      return { changed: false, added: 0, removed: 0, modified: 0, changes: [] };
    }
    
    const changes: string[] = [];
    let added = 0;
    let removed = 0;
    let modified = 0;
    
    // Simple line-by-line diff
    const maxLen = Math.max(oldLines.length, newLines.length);
    for (let i = 0; i < maxLen; i++) {
      const oldLine = oldLines[i];
      const newLine = newLines[i];
      
      if (oldLine === undefined && newLine !== undefined) {
        added++;
        changes.push(`+ ${newLine}`);
      } else if (newLine === undefined && oldLine !== undefined) {
        removed++;
        changes.push(`- ${oldLine}`);
      } else if (oldLine !== newLine) {
        modified++;
        changes.push(`- ${oldLine}`);
        changes.push(`+ ${newLine}`);
      }
    }
    
    return {
      changed: true,
      added,
      removed,
      modified,
      changes: changes.slice(0, 50), // Limit to first 50 changes
    };
  }
}

/**
 * Convenience function to run GEPA optimization
 */
export async function optimize(options: GEPAOptions): Promise<GEPAResult> {
  const gepa = new GEPA(options);
  return await gepa.optimize();
}