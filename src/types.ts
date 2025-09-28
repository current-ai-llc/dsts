import type { ZodSchema } from 'zod';
import { CoreMessage } from 'ai';

// Generic type parameters
export type DataInst = any;
export type Trajectory = any;
export type RolloutOutput = any;

// Core evaluation result
export interface EvaluationBatch<T = Trajectory, O = RolloutOutput> {
  outputs: O[];
  scores: number[]; // primary metric per instance (e.g., correctness)
  metrics?: Array<Record<string, number>>; // optional per-instance metrics (e.g., latency_ms)
  trajectories?: T[] | null;
}

// Candidate represents a set of text components
export type Candidate = Record<string, string>;

// GEPA Adapter interface - the main integration point
export interface GEPAAdapter<D = DataInst, T = Trajectory, O = RolloutOutput> {
  /**
   * Evaluate a candidate on a batch of inputs
   * @param batch List of inputs to evaluate
   * @param candidate Mapping of component names to text
   * @param captureTraces Whether to capture execution traces for reflection
   * @returns Evaluation results with outputs, scores, and optional traces
   */
  evaluate(
    batch: D[],
    candidate: Candidate,
    captureTraces?: boolean
  ): Promise<EvaluationBatch<T, O>>;

  /**
   * Build reflective dataset for component improvement
   * @param candidate Current candidate being evaluated
   * @param evalBatch Results from evaluation
   * @param componentsToUpdate Components that need updating
   * @returns Reflective dataset per component
   */
  makeReflectiveDataset(
    candidate: Candidate,
    evalBatch: EvaluationBatch<T, O>,
    componentsToUpdate: string[]
  ): Record<string, Array<Record<string, any>>>;

  /**
   * Optional custom proposal function
   */
  proposeNewTexts?: (
    candidate: Candidate,
    reflectiveDataset: Record<string, Array<Record<string, any>>>,
    componentsToUpdate: string[]
  ) => Promise<Candidate>;
}

// Language Model interface
export type LanguageModel = (prompt: string) => Promise<string>;
export type CostEstimator = (info: {
  model: string;
  input: string; // serialized user content
  output: string; // serialized output text/object
  result?: any; // raw result from AI SDK if available
}) => number; // USD cost for this single call
export type LanguageModelConfig = string | LanguageModel | {
  model: string;
  apiKey?: string;
  temperature?: number;
  // Optional per-call cost estimator used by the default adapter (task LM only)
  costEstimator?: CostEstimator;
  // Bounded concurrency for per-example calls of the task LM
  maxConcurrency?: number;
  // Tool passthrough for AI SDK generateText/generateObject
  tools?: Record<string, any>;
  // Provider-specific options (e.g., OpenAI reasoningEffort, serviceTier)
  providerOptions?: Record<string, any>;
  // Stop condition passthrough (e.g., ai.stepCountIs)
  stopWhen?: any;
  // Tool calling knobs
  toolChoice?: any;
  maxToolRoundtrips?: number;
  // Telemetry passthrough
  experimentalTelemetry?: Record<string, any>;
};

// Component selection strategies
export interface ComponentSelector {
  selectComponents(
    candidate: Candidate,
    iteration: number
  ): string[];
}

// Batch sampling strategies
export interface BatchSampler {
  nextBatch<T>(data: T[], iteration: number): T[];
}

// Candidate selection strategies
export interface CandidateSelector {
  selectCandidate(
    candidates: Array<{
      candidate: Candidate;
      scores: Record<string, number>;
      scalarScore: number;
    }>,
    paretoFront: number[]
  ): number;
}

// GEPA Options
export interface GEPAOptions {
  // Core settings
  seedCandidate: Candidate;
  trainset: DataInst[];
  valset?: DataInst[];
  adapter?: GEPAAdapter;
  
  // Model configuration
  taskLM?: LanguageModelConfig;
  reflectionLM?: LanguageModelConfig;
  // Provider-specific options for the reflection LM (passed to ai-sdk)
  reflectionLMProviderOptions?: Record<string, any>;
  
  // Strategy configuration
  candidateSelectionStrategy?: 'pareto' | 'current_best';
  componentSelector?: ComponentSelector | 'round_robin' | 'all';
  
  // Reflection settings
  skipPerfectScore?: boolean;
  reflectionMinibatchSize?: number;
  perfectScore?: number;
  /**
   * Optional short hint inserted into the reflection prompt to direct attention.
   * Keep this concise; it will be interpolated near the top of the prompt.
   */
  reflectionHint?: string;

  
  // Merge settings
  useMerge?: boolean;
  maxMergeInvocations?: number;
  
  // Budget
  maxMetricCalls?: number; // optional; if undefined, requires either maxIterations or maxBudgetUSD to be set
  maxIterations?: number; // optional hard cap on optimization iterations
  maxBudgetUSD?: number; // optional cost budget across all evaluations
  
  // Reproducibility
  seed?: number;
  
  // Logging
  logger?: Logger;
  displayProgressBar?: boolean;
  verbose?: boolean; // when true, print per-iteration prompts, scores, and summaries

  // Persistence (optional file-based)
  persistence?: {
    dir: string; // directory for checkpoint.json and archive.jsonl
    checkpointEveryIterations?: number; // default 1
    archiveFile?: string; // default 'archive.jsonl'
    resume?: boolean; // if true, try to resume from checkpoint.json
  };
}

// Logger interface
export interface Logger {
  log(level: string, message: string, data?: any): void;
}

// GEPA Result
export interface GEPAResult {
  bestCandidate: Candidate;
  bestScore: number;
  paretoFront: Array<{
    candidate: Candidate;
    scores: Record<string, number>;
    scalarScore: number;
  }>;
  iterations: number;
  totalMetricCalls: number;
  totalCostUSD: number;
  history: Array<{
    iteration: number;
    candidate: Candidate;
    scores: Record<string, number>;
    accepted: boolean;
  }>;
}

// For AI SDK integration
export interface GenerateObjectOptions<T> {
  model: string;
  messages: CoreMessage[];
  schema: ZodSchema<T>;
  system?: string;
  temperature?: number;
  maxRetries?: number;
}