import { ComponentSelector, BatchSampler, CandidateSelector, Candidate } from './types';
import { selectProgramCandidateFromInstanceFronts } from './pareto-utils';

/**
 * Round-robin component selector
 * Cycles through components one at a time
 */
export class RoundRobinComponentSelector implements ComponentSelector {
  selectComponents(candidate: Candidate, iteration: number): string[] {
    const components = Object.keys(candidate);
    if (components.length === 0) return [];
    const idx = iteration % components.length;
    return [components[idx]];
  }
}

/**
 * All components selector
 * Selects all components for modification
 */
export class AllComponentSelector implements ComponentSelector {
  selectComponents(candidate: Candidate, _iteration: number): string[] {
    return Object.keys(candidate);
  }
}

/**
 * Epoch-shuffled batch sampler
 * Shuffles data each epoch and returns minibatches
 */
export class EpochShuffledBatchSampler implements BatchSampler {
  private minibatchSize: number;
  private rng: () => number;
  private shuffled: number[] = [];
  private epoch: number = -1;
  private freq: Map<number, number> = new Map();

  constructor(minibatchSize: number, rng: () => number = Math.random) {
    this.minibatchSize = minibatchSize;
    this.rng = rng;
  }

  nextBatch<T>(data: T[], iteration: number): T[] {
    const trainSize = data.length;
    
    if (this.epoch === -1) {
      this.epoch = 0;
      this.updateShuffled(trainSize);
    }

    const mb = this.minibatchSize;
    const blocksPerEpoch = Math.max(1, Math.floor(this.shuffled.length / mb));
    const currEpoch = Math.floor(iteration / blocksPerEpoch);
    
    while (currEpoch >= this.epoch) {
      this.updateShuffled(trainSize);
    }

    const base = (iteration * mb) % this.shuffled.length;
    const indices = this.shuffled.slice(base, base + mb);
    
    return indices.map(i => data[i]);
  }

  private updateShuffled(trainSize: number): void {
    // Create array of indices
    const ids = Array.from({ length: trainSize }, (_, i) => i);
    
    // Fisher-Yates shuffle
    for (let i = ids.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }

    // Update frequency counts
    for (const i of ids) {
      this.freq.set(i, (this.freq.get(i) ?? 0) + 1);
    }

    // Pad to make divisible by minibatch size
    const mb = this.minibatchSize;
    const mod = trainSize % mb;
    const numToPad = mod === 0 ? 0 : mb - mod;
    
    if (numToPad > 0) {
      // Sort by frequency to pad with least-seen examples
      const candidates = Array.from({ length: trainSize }, (_, i) => i)
        .sort((a, b) => (this.freq.get(a) ?? 0) - (this.freq.get(b) ?? 0));
      
      const padded = [...ids];
      for (let k = 0; k < numToPad; k++) {
        const id = candidates[k % candidates.length];
        padded.push(id);
        this.freq.set(id, (this.freq.get(id) ?? 0) + 1);
      }
      
      this.shuffled = padded;
    } else {
      this.shuffled = ids;
    }
    
    this.epoch += 1;
  }
}

/**
 * Current best candidate selector
 * Always selects the candidate with highest scalar score
 */
export class CurrentBestCandidateSelector implements CandidateSelector {
  selectCandidate(
    candidates: Array<{
      candidate: Candidate;
      scores: Record<string, number>;
      scalarScore: number;
    }>,
    _paretoFront: number[]
  ): number {
    let bestIdx = 0;
    let bestScore = candidates[0].scalarScore;
    
    for (let i = 1; i < candidates.length; i++) {
      if (candidates[i].scalarScore > bestScore) {
        bestScore = candidates[i].scalarScore;
        bestIdx = i;
      }
    }
    
    return bestIdx;
  }
}

/**
 * Pareto-based candidate selector
 * Selects from Pareto front using instance-level performance
 */
export class ParetoCandidateSelector implements CandidateSelector {
  private rng: () => number;
  private instanceScores: number[][] = [];

  constructor(rng: () => number = Math.random) {
    this.rng = rng;
  }

  setInstanceScores(scores: number[][]): void {
    this.instanceScores = scores;
  }

  selectCandidate(
    candidates: Array<{
      candidate: Candidate;
      scores: Record<string, number>;
      scalarScore: number;
    }>,
    paretoFront: number[]
  ): number {
    if (this.instanceScores.length === 0) {
      // Fallback to best average score
      return new CurrentBestCandidateSelector().selectCandidate(candidates, paretoFront);
    }

    // Build instance fronts
    const nInst = this.instanceScores[0]?.length ?? 0;
    const instanceFronts: Array<Set<number>> = [];
    
    for (let i = 0; i < nInst; i++) {
      let best = Number.NEGATIVE_INFINITY;
      const front = new Set<number>();
      
      for (let k = 0; k < this.instanceScores.length; k++) {
        const v = this.instanceScores[k]?.[i] ?? 0;
        if (v > best) {
          best = v;
          front.clear();
          front.add(k);
        } else if (v === best) {
          front.add(k);
        }
      }
      
      instanceFronts.push(front);
    }

    // Get average scores per program
    const perProgScores = this.instanceScores.map(scores => {
      if (scores.length === 0) return 0;
      return scores.reduce((a, b) => a + b, 0) / scores.length;
    });

    return selectProgramCandidateFromInstanceFronts(
      instanceFronts,
      perProgScores,
      this.rng
    );
  }
}