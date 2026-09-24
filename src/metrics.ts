import { headingsMatch, normaliseHeading, type EvalQuery } from "./dataset";
import type { Candidate } from "./search";

/**
 * Two levels of ground truth, because they answer different questions.
 *
 * `doc` — did retrieval find the right document at all? This is the level the
 * the private corpus ground truth was written at, and the level a citation needs.
 *
 * `section` — did it find the right paragraph inside it? The ground truth names
 * the section heading, and since these documents run to several hundred chunks
 * each, `doc` alone can look healthy while the answer is still not in context.
 */
export type Level = "doc" | "section";

export function isRelevant(c: Candidate, q: EvalQuery, level: Level): boolean {
  if (!q.files.includes(c.docFile)) return false;
  if (level === "doc") return true;
  if (q.sections.length === 0) return false;
  const path = normaliseHeading(c.headingPath);
  return q.sections.some((s) => {
    if (s.trim().length === 0) return false;
    if (headingsMatch(s, c.heading)) return true;
    // A windowed chunk deep inside a section keeps the whole heading path, so a
    // long gold heading can still be found there verbatim.
    const want = normaliseHeading(s);
    return want.length > 12 && path.includes(want);
  });
}

export interface QueryMetrics {
  /** 1-based rank of the first relevant candidate, or null if none in the list. */
  firstRelevantRank: number | null;
  recallAt: Record<number, number>;
  reciprocalRank: number;
  ndcg: number;
}

const DEPTHS = [1, 3, 5, 10] as const;

export function scoreQuery(ranked: Candidate[], q: EvalQuery, level: Level, evalK: number): QueryMetrics {
  const rel: number[] = ranked.map((c) => (isRelevant(c, q, level) ? 1 : 0));
  const first = rel.findIndex((r) => r === 1);
  const firstRelevantRank = first === -1 ? null : first + 1;

  const recallAt: Record<number, number> = {};
  for (const d of DEPTHS) recallAt[d] = rel.slice(0, d).some((r) => r === 1) ? 1 : 0;

  const reciprocalRank = firstRelevantRank !== null && firstRelevantRank <= evalK ? 1 / firstRelevantRank : 0;

  // Binary-gain nDCG at evalK. The ideal list puts every relevant candidate
  // present in the shortlist at the top, so a query whose gold passage the
  // first stage never retrieved scores 0 rather than being excluded — the
  // reranker genuinely cannot fix that case and the number should say so.
  let dcg = 0;
  for (let i = 0; i < Math.min(evalK, rel.length); i++) if (rel[i] === 1) dcg += 1 / Math.log2(i + 2);
  const totalRelevant = rel.reduce((a, b) => a + b, 0);
  let idcg = 0;
  for (let i = 0; i < Math.min(evalK, totalRelevant); i++) idcg += 1 / Math.log2(i + 2);
  const ndcg = idcg > 0 ? dcg / idcg : 0;

  return { firstRelevantRank, recallAt, reciprocalRank, ndcg };
}

export interface Aggregate {
  n: number;
  recallAt: Record<number, number>;
  mrr: number;
  ndcg: number;
}

export function aggregate(rows: QueryMetrics[]): Aggregate {
  const n = rows.length || 1;
  const recallAt: Record<number, number> = {};
  for (const d of DEPTHS) recallAt[d] = rows.reduce((a, r) => a + (r.recallAt[d] ?? 0), 0) / n;
  return {
    n: rows.length,
    recallAt,
    mrr: rows.reduce((a, r) => a + r.reciprocalRank, 0) / n,
    ndcg: rows.reduce((a, r) => a + r.ndcg, 0) / n,
  };
}

/**
 * The most important number in the whole report.
 *
 * A reranker only reorders the shortlist it is given, so no reranking strategy
 * can score above the fraction of queries whose shortlist contains a relevant
 * passage at all. Every gain has to be read against this ceiling.
 */
export function shortlistCeiling(shortlists: { q: EvalQuery; candidates: Candidate[] }[], level: Level): number {
  const hits = shortlists.filter(({ q, candidates }) => candidates.some((c) => isRelevant(c, q, level)));
  return hits.length / (shortlists.length || 1);
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[i]!;
}

/**
 * Which queries can be scored at section level at all.
 *
 * 12 of the 84 rows name a heading the document does not contain — the guide
 * documents are cited as "Luku 1.1 Palkkakäsite" where the markdown has no such
 * heading, and the KVTES file has no internal headings to match. Those queries
 * score zero at section level for every strategy alike, so including them only
 * drags every absolute number down by the same amount while measuring the
 * heading matcher rather than retrieval. They are excluded and counted.
 */
export function sectionEvaluable(queries: EvalQuery[], chunks: Candidate[]): boolean[] {
  const byDoc = new Map<string, Candidate[]>();
  for (const c of chunks) {
    const list = byDoc.get(c.docFile);
    if (list) list.push(c);
    else byDoc.set(c.docFile, [c]);
  }
  return queries.map((q) => q.files.flatMap((f) => byDoc.get(f) ?? []).some((c) => isRelevant(c, q, "section")));
}
