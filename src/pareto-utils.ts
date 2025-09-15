/**
 * Pareto utilities for multi-objective optimization
 * Based on the ax implementation
 */

export interface ParetoPoint {
  idx: number;
  scores: Record<string, number>;
  dominated?: number[];
}

/**
 * Build Pareto frontier from candidates
 */
export function buildParetoFront(
  candidates: Array<{ idx: number; scores: Record<string, number> }>,
  tieEpsilon: number = 0
): ParetoPoint[] {
  const paretoFront: ParetoPoint[] = [];
  
  for (const candidate of candidates) {
    let isDominated = false;
    const dominatedIndices: number[] = [];
    
    // Check if this candidate is dominated by any in the current front
    for (let i = 0; i < paretoFront.length; i++) {
      const frontPoint = paretoFront[i];
      const dominance = checkDominance(
        candidate.scores,
        frontPoint.scores,
        tieEpsilon
      );
      
      if (dominance === -1) {
        // candidate is dominated
        isDominated = true;
        break;
      } else if (dominance === 1) {
        // candidate dominates frontPoint
        dominatedIndices.push(i);
      }
    }
    
    if (!isDominated) {
      // Remove dominated points from front
      const newFront = paretoFront.filter((_, i) => !dominatedIndices.includes(i));
      newFront.push({
        idx: candidate.idx,
        scores: candidate.scores,
        dominated: dominatedIndices.map(i => paretoFront[i].idx)
      });
      paretoFront.length = 0;
      paretoFront.push(...newFront);
    }
  }
  
  return paretoFront;
}

/**
 * Check dominance between two score vectors
 * Returns: 1 if a dominates b, -1 if b dominates a, 0 if neither
 */
function checkDominance(
  a: Record<string, number>,
  b: Record<string, number>,
  epsilon: number
): number {
  // Use union of keys to compare on the same objective space
  const keys = new Set<string>([...Object.keys(a), ...Object.keys(b)]);
  let aDominates = false;
  let bDominates = false;
  
  for (const key of keys) {
    const aVal = a[key] ?? 0;
    const bVal = b[key] ?? 0;
    
    if (aVal > bVal + epsilon) {
      aDominates = true;
    } else if (bVal > aVal + epsilon) {
      bDominates = true;
    }
  }
  
  if (aDominates && !bDominates) return 1;
  if (bDominates && !aDominates) return -1;
  return 0;
}

/**
 * Calculate 2D hypervolume (area under Pareto front)
 * Assumes maximization objectives
 */
export function hypervolume2D(
  points: Array<Record<string, number>>,
  referencePoint?: Record<string, number>
): number | null {
  if (points.length === 0) return 0;
  
  // Get objective names
  const keys = Object.keys(points[0]);
  if (keys.length !== 2) return null; // Only works for 2D
  
  const [obj1, obj2] = keys;
  
  // Use reference point or find minimum values
  const ref1 = referencePoint?.[obj1] ?? Math.min(...points.map(p => p[obj1] ?? 0)) - 1;
  const ref2 = referencePoint?.[obj2] ?? Math.min(...points.map(p => p[obj2] ?? 0)) - 1;
  
  // Sort points by first objective (descending)
  const sorted = [...points].sort((a, b) => (b[obj1] ?? 0) - (a[obj1] ?? 0));
  
  let volume = 0;
  let prevY = ref2;
  
  for (const point of sorted) {
    const x = point[obj1] ?? 0;
    const y = point[obj2] ?? 0;
    
    if (x > ref1 && y > prevY) {
      volume += (x - ref1) * (y - prevY);
      prevY = y;
    }
  }
  
  return volume;
}

/**
 * Calculate average of array
 */
export function average(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/**
 * Average vector of scores
 */
export function avgVec(vecs: Array<Record<string, number>>): Record<string, number> {
  if (vecs.length === 0) return {};
  
  const result: Record<string, number> = {};
  const keys = new Set<string>();
  
  for (const vec of vecs) {
    for (const key of Object.keys(vec)) {
      keys.add(key);
    }
  }
  
  for (const key of keys) {
    const values = vecs.map(v => v[key] ?? 0);
    result[key] = average(values);
  }
  
  return result;
}

/**
 * Select program candidate from instance fronts
 * Based on Algorithm 2 from the paper
 */
export function selectProgramCandidateFromInstanceFronts(
  instanceFronts: Array<Set<number>>,
  perProgScores: number[],
  rand: () => number = Math.random
): number {
  // Count frequency of each program in instance fronts
  const freq = new Map<number, number>();
  
  for (const front of instanceFronts) {
    for (const prog of front) {
      freq.set(prog, (freq.get(prog) ?? 0) + 1);
    }
  }
  
  // Build cumulative distribution
  const candidates = Array.from(freq.keys());
  const weights = candidates.map(c => freq.get(c)!);
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  
  if (totalWeight === 0 || candidates.length === 0) {
    // Fallback to best average score
    let bestIdx = 0;
    let bestScore = perProgScores[0];
    for (let i = 1; i < perProgScores.length; i++) {
      if (perProgScores[i] > bestScore) {
        bestScore = perProgScores[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  
  // Sample from distribution
  let r = rand() * totalWeight;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r <= 0) {
      return candidates[i];
    }
  }
  
  return candidates[candidates.length - 1];
}

/**
 * Remove dominated programs by instance fronts
 */
export function removeDominatedProgramsByInstanceFronts(
  instanceFronts: Array<Set<number>>,
  perProgScores: number[]
): Array<Set<number>> {
  const allProgs = new Set<number>();
  for (const front of instanceFronts) {
    for (const prog of front) {
      allProgs.add(prog);
    }
  }
  
  // For each program, check if it appears in at least one front
  const nonDominated = new Set<number>();
  for (const prog of allProgs) {
    for (const front of instanceFronts) {
      if (front.has(prog)) {
        nonDominated.add(prog);
        break;
      }
    }
  }
  
  // Filter fronts to only include non-dominated programs
  return instanceFronts.map(front => {
    const filtered = new Set<number>();
    for (const prog of front) {
      if (nonDominated.has(prog)) {
        filtered.add(prog);
      }
    }
    return filtered;
  });
}