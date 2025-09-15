export { GEPA, optimize } from './gepa';
export { DefaultAdapter } from './adapters/default-adapter';
export type { DefaultAdapterOptions, DefaultAdapterTask, DefaultAdapterTrace } from './adapters/default-adapter';

export * from './types';
export * from './pareto-utils';
export * from './strategies';

// Re-export commonly used items for convenience
export type {
  GEPAOptions,
  GEPAResult,
  GEPAAdapter,
  Candidate,
  EvaluationBatch,
  ComponentSelector,
  BatchSampler,
  CandidateSelector,
  Logger,
  LanguageModel,
  LanguageModelConfig,
} from './types';