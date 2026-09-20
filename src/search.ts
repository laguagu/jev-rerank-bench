import { CONFIG } from "./config";
import { sql } from "./db";
import { toPgVector } from "./embed";

export interface Candidate {
  chunkId: number;
  docFile: string;
  docTitle: string;
  heading: string;
  headingPath: string;
  body: string;
  /** Strategy-native score: ts_rank_cd, cosine similarity, or an RRF sum. */
  score: number;
}

export interface SearchResult {
  candidates: Candidate[];
  /**
   * Wall-clock milliseconds, including the round trip to the database host.
   * Against a hosted Postgres this is dominated by the network, so it is a
   * budget for the search step, not a measurement of pgvector.
   */
  wallMs: number;
}

/**
 * Turn a natural-language question into a `websearch_to_tsquery` input whose
 * terms are OR-ed.
 *
 * Done here rather than with `regexp_replace` in the SQL: postgres.js reads the
 * *cooked* template string, so a `'\s+'` written in a tagged template arrives at
 * Postgres as `'s+'` and the pattern silently replaces every run of the letter
 * s. The bug does not throw — it just ruins the lexical baseline.
 *
 * A leading `-` is stripped because websearch reads it as NOT, and a stray
 * double quote is dropped because it opens a phrase that never closes.
 */
export function orQuery(question: string): string {
  return question
    .replace(/["]/g, " ")
    .split(/\s+/)
    .map((w) => w.replace(/^[-+]+/, "").trim())
    .filter((w) => w.length > 0 && w.toLowerCase() !== "or" && w.toLowerCase() !== "and")
    .join(" OR ");
}

/**
 * Lexical retrieval over Postgres' built-in `finnish` configuration.
 *
 * The terms are OR-ed rather than AND-ed: these questions are full sentences
 * ("Kuinka paljon ikääntyneen työntekijän työaikaa lyhennetään lentoliikenteen
 * palvelualan TES:n mukaan?") and requiring every lexeme returns nothing on
 * most of them. `ts_rank_cd` then rewards the passages covering more of the
 * query. This is Postgres FTS, not BM25 — Neon has no `pg_search`.
 */
export async function ftsSearch(query: string, k = CONFIG.candidateK): Promise<SearchResult> {
  const t = Date.now();
  const rows = await sql<Candidate[]>`
    WITH q AS (
      SELECT websearch_to_tsquery('finnish', ${orQuery(query)}) AS tsq
    )
    SELECT c.id AS "chunkId", c.doc_file AS "docFile", d.title AS "docTitle",
           c.heading, c.heading_path AS "headingPath", c.body,
           ts_rank_cd(c.tsv, q.tsq)::float8 AS score
    FROM chunks c JOIN documents d ON d.file = c.doc_file, q
    WHERE c.tsv @@ q.tsq
    ORDER BY score DESC, c.id
    LIMIT ${k}`;
  return { candidates: rows, wallMs: Date.now() - t };
}

/** Dense retrieval over the HNSW index. `<=>` is cosine distance. */
export async function vectorSearch(queryVector: number[], k = CONFIG.candidateK): Promise<SearchResult> {
  const t = Date.now();
  const v = toPgVector(queryVector);
  const rows = await sql<Candidate[]>`
    SELECT c.id AS "chunkId", c.doc_file AS "docFile", d.title AS "docTitle",
           c.heading, c.heading_path AS "headingPath", c.body,
           (1 - (c.embedding <=> ${v}::vector))::float8 AS score
    FROM chunks c JOIN documents d ON d.file = c.doc_file
    ORDER BY c.embedding <=> ${v}::vector
    LIMIT ${k}`;
  return { candidates: rows, wallMs: Date.now() - t };
}

/**
 * Reciprocal rank fusion of the two lists.
 *
 * Both sides are over-fetched to `k` before fusion, otherwise a passage that is
 * rank 25 lexically and rank 3 densely never reaches the fused list. Score
 * scales are never compared — only ranks — which is the whole point of RRF when
 * `ts_rank_cd` and cosine similarity have no common unit.
 */
export async function hybridSearch(
  query: string,
  queryVector: number[],
  k = CONFIG.candidateK,
  fetchDepth = CONFIG.candidateK * 2,
): Promise<SearchResult> {
  const t = Date.now();
  const v = toPgVector(queryVector);
  const rows = await sql<Candidate[]>`
    WITH q AS (
      SELECT websearch_to_tsquery('finnish', ${orQuery(query)}) AS tsq
    ),
    -- The window function ranks the already-limited subquery rather than the
    -- whole table. Written the other way round — row_number() OVER (ORDER BY
    -- embedding <=> v) beside an ORDER BY ... LIMIT — the window has to see
    -- every qualifying row before the limit applies.
    --
    -- Measured on this corpus (37 440 rows, EXPLAIN ANALYZE, warm): 394 ms for
    -- the window-inside-LIMIT form against 314 ms for this one. Real but
    -- modest, and *not* the difference between using the HNSW index and not —
    -- see scripts/check-index-use.ts, which shows the planner declining the
    -- index for both shapes at this size, correctly. A backtick in this comment
    -- would end the template literal the query lives in; keep them out.
    lexical AS (
      SELECT id, row_number() OVER () AS rank FROM (
        SELECT c.id
        FROM chunks c, q
        WHERE c.tsv @@ q.tsq
        ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC, c.id
        LIMIT ${fetchDepth}
      ) t
    ),
    dense AS (
      SELECT id, row_number() OVER () AS rank FROM (
        SELECT c.id
        FROM chunks c
        ORDER BY c.embedding <=> ${v}::vector
        LIMIT ${fetchDepth}
      ) t
    ),
    fused AS (
      SELECT id, sum(w)::float8 AS score FROM (
        SELECT id, 1.0 / (${CONFIG.rrfK} + rank) AS w FROM lexical
        UNION ALL
        SELECT id, 1.0 / (${CONFIG.rrfK} + rank) AS w FROM dense
      ) x GROUP BY id
    )
    SELECT c.id AS "chunkId", c.doc_file AS "docFile", d.title AS "docTitle",
           c.heading, c.heading_path AS "headingPath", c.body,
           f.score
    FROM fused f
    JOIN chunks c ON c.id = f.id
    JOIN documents d ON d.file = c.doc_file
    ORDER BY f.score DESC, c.id
    LIMIT ${k}`;
  return { candidates: rows, wallMs: Date.now() - t };
}

/** What a reranker and a metric both read as "the passage". */
export function passageText(c: Candidate): string {
  const head = [c.docTitle, c.headingPath].filter(Boolean).join(" — ");
  return head ? `${head}\n\n${c.body}` : c.body;
}
