import { generateObject, generateText } from 'ai';
import type { ZodSchema } from 'zod';
import { normalizeUsage, estimateCost } from 'tokenlens';
import { GEPAAdapter, EvaluationBatch, Candidate } from '../types';

export interface DefaultAdapterOptions {
  model: string;
  apiKey?: string;
  temperature?: number;
  maxRetries?: number;
  // Optional per-call cost estimator; when provided, records { cost_usd } in metrics
  costEstimator?: (info: { model: string; input: string; output: string; result?: any }) => number;
  // Bounded concurrency for per-example calls (default: 1)
  maxConcurrency?: number;
  // Tool and provider passthrough
  tools?: Record<string, any>;
  providerOptions?: Record<string, any>;
  stopWhen?: any;
  toolChoice?: any;
  maxToolRoundtrips?: number;
  experimentalTelemetry?: Record<string, any>;
}

export interface DefaultAdapterTask<T = any> {
  input: any;
  expectedOutput?: T;
  schema?: ZodSchema<T>;
  scorer?: (prediction: T, expected?: T) => number;
}

export interface DefaultAdapterTrace {
  input: any;
  systemPrompt: string;
  userPrompt: string;
  output: any;
  error?: string;
  score: number;
  expectedOutput?: any;
  latencyMs?: number;
}

/**
 * Default adapter for AI SDK generateObject/generateText
 * Supports both structured (with schema) and unstructured generation
 */
export class DefaultAdapter<T = any> implements GEPAAdapter<DefaultAdapterTask<T>, DefaultAdapterTrace, T> {
  private model: string;
  private apiKey?: string;
  private temperature: number;
  private maxRetries: number;
  private costEstimator?: (info: { model: string; input: string; output: string; result?: any }) => number;
  private maxConcurrency: number;
  private tools?: Record<string, any>;
  private providerOptions?: Record<string, any>;
  private stopWhen?: any;
  private toolChoice?: any;
  private maxToolRoundtrips?: number;
  private experimentalTelemetry?: Record<string, any>;
  
  constructor(options: DefaultAdapterOptions) {
    this.model = options.model;
    this.apiKey = options.apiKey;
    this.temperature = options.temperature ?? 0.7;
    this.maxRetries = options.maxRetries ?? 3;
    this.costEstimator = options.costEstimator ?? this.buildDefaultCostEstimator(options.model);
    this.maxConcurrency = Math.max(1, options.maxConcurrency ?? 10);
    this.tools = options.tools;
    this.providerOptions = options.providerOptions;
    this.stopWhen = options.stopWhen;
    this.toolChoice = options.toolChoice;
    this.maxToolRoundtrips = options.maxToolRoundtrips;
    this.experimentalTelemetry = options.experimentalTelemetry;
  }

  private buildDefaultCostEstimator(model: string) {
    // Prefer TokenLens: use usage from AI SDK when available
    const toModelId = (m: string) => m.replace('/', ':');
    return ({ input, output, result }: { model: string; input: string; output: string; result?: any }) => {
      try {
        const usage = (result as any)?.usage || (result as any)?.response?.usage;
        if (usage) {
          const u = normalizeUsage(usage);
          const est = estimateCost({ modelId: toModelId(model), usage: u });
          return est.totalUSD ?? 0;
        }
      } catch {}
      // Fallback: rough char-based estimate if usage not provided
      const PRICE: Record<string, { in: number; out: number }> = {
        'openai/gpt-4o-mini': { in: 0.15 / 1e6, out: 0.60 / 1e6 },
        'openai/gpt-4o': { in: 5 / 1e6, out: 15 / 1e6 },
      };
      const p = PRICE[model] || PRICE['openai/gpt-4o-mini'];
      const estTokens = (s: string) => Math.ceil((s?.length ?? 0) / 4);
      const inTok = estTokens(input);
      const outTok = estTokens(output);
      return inTok * p.in + outTok * p.out;
    };
  }

  async evaluate(
    batch: DefaultAdapterTask<T>[],
    candidate: Candidate,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<DefaultAdapterTrace, T>> {
    const n = batch.length;
    const outputs = new Array<T>(n);
    const scores: number[] = new Array(n);
    const metrics: Array<Record<string, number>> = new Array(n);
    const trajectories: DefaultAdapterTrace[] | null = captureTraces ? new Array(n) : null;

    // Get system prompt from candidate (default component name)
    const systemPrompt = candidate['system'] || candidate['systemPrompt'] || candidate['instruction'] || '';

    const runOne = async (i: number) => {
      const task = batch[i];
      const t0 = Date.now();
      let latencyMs = 0;
      try {
        // Convert input to string if needed
        const userPrompt = typeof task.input === 'string' ? task.input : JSON.stringify(task.input);

        let rawResult: any;
        let output: T;
        const commonOpts = {
          temperature: this.temperature,
          maxRetries: this.maxRetries,
          tools: this.tools,
          providerOptions: this.providerOptions,
          stopWhen: this.stopWhen,
          toolChoice: this.toolChoice,
          maxToolRoundtrips: this.maxToolRoundtrips,
          experimental_telemetry: this.experimentalTelemetry,
        } as any;

        if (task.schema) {
          const result = await generateObject({
            model: this.model,
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userPrompt },
            ],
            schema: task.schema,
            ...commonOpts,
          });
          rawResult = result;
          output = result.object as T;
        } else {
          const result = await generateText({
            model: this.model,
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userPrompt },
            ],
            ...commonOpts,
          });
          rawResult = result;
          output = result.text as any as T;
        }

        latencyMs = Date.now() - t0;

        // Optional cost estimation
        let costUSD = 0;
        if (this.costEstimator) {
          const outStr = typeof output === 'string' ? output : JSON.stringify(output);
          const combinedInput = `${systemPrompt}\n${userPrompt}`;
          try {
            costUSD = this.costEstimator({ model: this.model, input: combinedInput, output: outStr, result: rawResult });
          } catch {}
        }

        // Calculate score
        let score = 0;
        if (task.scorer) {
          score = task.scorer(output, task.expectedOutput);
        } else if (task.expectedOutput !== undefined) {
          score = this.defaultScorer(output, task.expectedOutput);
        } else {
          score = 1;
        }

        outputs[i] = output;
        scores[i] = score;
        metrics[i] = { latency_ms: latencyMs, cost_usd: costUSD };
        if (trajectories) {
          trajectories[i] = {
            input: task.input,
            systemPrompt,
            userPrompt,
            output,
            score,
            expectedOutput: task.expectedOutput,
            latencyMs,
          };
        }
      } catch (error) {
        latencyMs = Date.now() - t0;
        outputs[i] = {} as T;
        scores[i] = 0;
        metrics[i] = { latency_ms: latencyMs, cost_usd: 0 };
        if (trajectories) {
          const userPrompt = typeof batch[i].input === 'string' ? batch[i].input : JSON.stringify(batch[i].input);
          trajectories[i] = {
            input: batch[i].input,
            systemPrompt,
            userPrompt,
            output: null,
            error: error instanceof Error ? error.message : String(error),
            score: 0,
            expectedOutput: batch[i].expectedOutput,
            latencyMs,
          };
        }
      }
    };

    // Bounded concurrency workers
    let cursor = 0;
    const max = this.maxConcurrency;
    const worker = async () => {
      while (true) {
        const i = cursor++;
        if (i >= n) break;
        await runOne(i);
      }
    };
    const workers = Array.from({ length: max }, () => worker());
    await Promise.all(workers);

    return {
      outputs,
      scores,
      metrics,
      trajectories: captureTraces ? (trajectories as DefaultAdapterTrace[]) : null,
    };
  }

  makeReflectiveDataset(
    candidate: Candidate,
    evalBatch: EvaluationBatch<DefaultAdapterTrace, T>,
    componentsToUpdate: string[]
  ): Record<string, Array<Record<string, any>>> {
    const dataset: Record<string, Array<Record<string, any>>> = {};

    if (!evalBatch.trajectories) {
      return dataset;
    }

    // For each component to update
    for (const component of componentsToUpdate) {
      const examples: Array<Record<string, any>> = [];

      // Collect examples from trajectories
      for (let i = 0; i < evalBatch.trajectories.length; i++) {
        const trace = evalBatch.trajectories[i];
        const score = evalBatch.scores[i];

        // Focus on failed or low-scoring examples
        if (score < 0.9) {
          const example: Record<string, any> = {
            Inputs: {
              userMessage: trace.userPrompt,
              systemPrompt: trace.systemPrompt,
            },
            'Generated Outputs': trace.output || trace.error || 'No output',
            Feedback: this.generateFeedback(trace, score),
          };

          examples.push(example);
        }
      }

      // If no failures, include a sample of successes for context
      if (examples.length === 0 && evalBatch.trajectories.length > 0) {
        const numExamples = Math.min(3, evalBatch.trajectories.length);
        for (let i = 0; i < numExamples; i++) {
          const trace = evalBatch.trajectories[i];
          const score = evalBatch.scores[i];
          
          examples.push({
            Inputs: {
              userMessage: trace.userPrompt,
              systemPrompt: trace.systemPrompt,
            },
            'Generated Outputs': trace.output,
            Feedback: `Successful execution with score ${score}`,
          });
        }
      }

      dataset[component] = examples;
    }

    return dataset;
  }

  private defaultScorer(prediction: any, expected: any): number {
    // Deep equality check for objects
    if (typeof prediction === 'object' && typeof expected === 'object') {
      return JSON.stringify(prediction) === JSON.stringify(expected) ? 1 : 0;
    }
    return prediction === expected ? 1 : 0;
  }

  private generateFeedback(trace: DefaultAdapterTrace, score: number): string {
    const parts: string[] = [];

    if (trace.error) {
      parts.push(`Error: ${trace.error}`);
    }

    if (trace.expectedOutput !== undefined) {
      parts.push(`Expected: ${JSON.stringify(trace.expectedOutput)}`);
      if (trace.output !== null && trace.output !== undefined) {
        parts.push(`Got: ${JSON.stringify(trace.output)}`);
      }
    }

    parts.push(`Score: ${score}`);

    if (score === 0 && !trace.error) {
      parts.push('The output did not match the expected result.');
    }

    return parts.join(' | ');
  }
}