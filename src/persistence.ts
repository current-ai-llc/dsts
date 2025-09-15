import fs from 'fs';
import path from 'path';
import { Candidate } from './types';

export interface CheckpointState {
  iteration: number;
  totalMetricCalls: number;
  totalCostUSD: number;
  rngState: number;
  candidates: Array<{
    candidate: Candidate;
    scores: Record<string, number>;
    scalarScore: number;
    parent?: number;
  }>;
  perInstanceScores: number[][];
}

export interface ArchiveRecord {
  ts: number;
  iteration: number;
  event: 'start' | 'accepted' | 'rejected' | 'stagnation' | 'finish';
  data?: any;
}

export class FilePersistence {
  private dir: string;
  private archivePath: string;
  private checkpointPath: string;

  constructor(dir: string, archiveFile: string = 'archive.jsonl') {
    this.dir = dir;
    this.archivePath = path.join(dir, archiveFile);
    this.checkpointPath = path.join(dir, 'checkpoint.json');
  }

  ensureDir(): void {
    if (!fs.existsSync(this.dir)) {
      fs.mkdirSync(this.dir, { recursive: true });
    }
  }

  saveCheckpoint(state: CheckpointState): void {
    this.ensureDir();
    fs.writeFileSync(this.checkpointPath, JSON.stringify(state, null, 2), 'utf8');
  }

  loadCheckpoint(): CheckpointState | null {
    try {
      if (!fs.existsSync(this.checkpointPath)) return null;
      const txt = fs.readFileSync(this.checkpointPath, 'utf8');
      return JSON.parse(txt) as CheckpointState;
    } catch {
      return null;
    }
  }

  appendArchive(rec: ArchiveRecord): void {
    this.ensureDir();
    fs.appendFileSync(this.archivePath, JSON.stringify(rec) + '\n', 'utf8');
  }
}
