/**
 * All tunables in one place. The benchmark is only meaningful if every strategy
 * sees the identical corpus, the identical queries and the identical candidate
 * lists, so these constants are read from here and never redefined locally.
 */

/** `--candidates=100` or `CANDIDATES=100` overrides the shortlist depth for one
 *  run. The depth is the reranker's ceiling, so sweeping it is the only way to
 *  separate "the reranker is wrong" from "the answer was never in the list". */
function numArg(flag: string, envVar: string, fallback: number): number {
  const fromArg = process.argv.find((a) => a.startsWith(`--${flag}=`))?.split("=")[1];
  const v = Number(fromArg ?? process.env[envVar] ?? NaN);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

const CANDIDATE_K = numArg("candidates", "CANDIDATES", 30);

export const CONFIG = {
  /** Documents in the indexed corpus. All ground-truth documents are always
   *  included; the rest are distractors drawn from the same folders. Finnish
   *  collective agreements repeat each other almost section for section, so the
   *  distractors are what makes the task hard. */
  corpusDocs: 160,

  /** Chunking. Sections are split on markdown headings first, then windowed. */
  chunkMaxChars: 1400,
  chunkOverlapChars: 200,
  /** A heading with less text than this is merged into the next chunk rather
   *  than indexed on its own — a lone "## 12 §" row retrieves nothing useful. */
  chunkMinChars: 120,

  /** Embeddings. 1536 dims matches the private corpus production table, and pgvector
   *  indexes up to 2000 dims with HNSW. */
  embeddingDims: 1536,
  embeddingBatchSize: 64,

  /** Retrieval. `candidateK` is the shortlist every reranker receives;
   *  `evalK` is the depth the metrics are computed at. */
  candidateK: CANDIDATE_K,
  evalK: 10,
  /** RRF constant. 60 is the value from the original Cormack et al. paper and
   *  the one pgvector hybrid examples use. */
  rrfK: 60,

  /** HNSW search breadth. Must comfortably exceed the deepest LIMIT any query
   *  asks for — the hybrid query fetches `candidateK * 2` dense rows — or the
   *  index returns a short, quietly worse list. pgvector defaults this to 40,
   *  which is below even the k=30 hybrid depth. Derived from the depth so a
   *  `--candidates=100` sweep does not silently degrade the dense side. */
  hnswEfSearch: Math.max(200, CANDIDATE_K * 4),

  /** Jev. 12 concurrent requests is what the TypeSafe rerank cookbook uses and
   *  sits far under the documented 1200 req/min ceiling. */
  jevConcurrency: 12,
  jevModel: process.env.TYPESAFE_DEFAULT_MODEL ?? "jev-latest",
  /** $ per million input tokens for jev-1.13; output is free. */
  jevInputCostPerMillion: 0.042,

  /** Deterministic sampling. */
  seed: 20260920,
} as const;

export type Config = typeof CONFIG;
