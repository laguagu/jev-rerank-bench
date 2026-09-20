import { existsSync, readFileSync } from "node:fs";
import { DATASET } from "../dataset-config";
import type { Candidate } from "../search";
import { order, type RerankOutcome } from "./jev";

/**
 * A reranker whose scores were computed elsewhere and written to disk.
 *
 * Laya runs in Python from local weights, so it cannot be called from the
 * benchmark the way a hosted API can. `scripts/laya_score.py` scores every
 * (query, candidate) pair against the same question spec and writes the result;
 * this reads it back so the open-weight model appears in the same tables, under
 * the same metrics, over the same shortlists as everything else.
 *
 * Latency is reported from the Python run rather than measured here — replaying
 * a JSON file would report microseconds and mean nothing.
 */
export interface PrecomputedScores {
  kind: string;
  subfolder?: string;
  perQueryMsP50: number;
  perQueryMsP95: number;
  failures: number;
  scores: Record<string, Record<string, number>>;
}

export function precomputedPath(name: string): string {
  return `${DATASET.dir}/${name}.json`;
}

export function loadPrecomputed(name: string): PrecomputedScores | null {
  const p = precomputedPath(name);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf8")) as PrecomputedScores;
}

/**
 * Returns a reranker closure over the loaded table. A query id missing from the
 * table leaves its shortlist untouched rather than being scored as zero, so a
 * partial run degrades to "no reranking" instead of to a shuffled list.
 */
export function precomputedRerank(table: PrecomputedScores, queryIds: string[]) {
  let index = 0;
  return async (_question: string, candidates: Candidate[]): Promise<RerankOutcome> => {
    const qid = queryIds[index++]!;
    const row = table.scores[qid];
    const scores = new Map<number, number>();
    let failed = 0;
    for (const c of candidates) {
      const v = row?.[String(c.chunkId)];
      if (typeof v === "number" && Number.isFinite(v)) scores.set(c.chunkId, v);
      else failed++;
    }
    return {
      ranked: row ? order(candidates, scores) : candidates,
      scores,
      inputTokens: 0,
      outputTokens: 0,
      requests: 1,
      failed,
    };
  };
}
