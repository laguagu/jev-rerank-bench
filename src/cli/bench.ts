/**
 * The benchmark.
 *
 * Every reranking strategy is handed the *same* shortlist, produced once per
 * first-stage retriever and cached, so a difference in the metrics can only come
 * from the ordering. Retrieval is measured separately from reranking: the first
 * stage is one database round trip, the rerankers are network calls to a model,
 * and adding them together would hide which one costs what.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { CONFIG } from "../config";
import { DATASET } from "../dataset-config";
import { close } from "../db";
import type { EvalQuery } from "../dataset";
import { embedAll } from "../embed";
import { aggregate, percentile, scoreQuery, sectionEvaluable, shortlistCeiling, type Aggregate, type Level } from "../metrics";
import { jevRerankBatched, jevRerankCascade, jevRerankPerCall, jevRerankScore, type RerankOutcome } from "../rerank/jev";
import { llmConfigured, llmRerank, LLM_VARIANTS } from "../rerank/llm";
import { loadPrecomputed, precomputedRerank } from "../rerank/precomputed";
import { voyageConfigured, voyageRerank } from "../rerank/voyage";
import { ftsSearch, hybridSearch, passageText, vectorSearch, type Candidate } from "../search";

const only = process.argv.slice(2).filter((a) => !a.startsWith("-"));
const wanted = (name: string) => only.length === 0 || only.includes(name);

const corpus = JSON.parse(readFileSync(`${DATASET.dir}/corpus.json`, "utf8")) as { queries: EvalQuery[]; chunks: { docFile: string; docTitle: string; heading: string; headingPath: string; text: string }[] };
// `--limit N` runs a smoke test against the first N queries without touching
// the cached shortlists, which stay keyed to the full set.
const limitArg = process.argv.find((a) => a.startsWith("--limit="));
const limit = limitArg ? Number(limitArg.split("=")[1]) : corpus.queries.length;
const queries = corpus.queries.slice(0, limit);
// A smoke run must not become the record. `--limit` once overwrote a full
// 13-strategy result file with three queries and two strategies.
if (limitArg && !process.argv.some((a) => a.startsWith("--tag="))) {
  process.argv.push("--tag=smoke");
}

const sectionMask = sectionEvaluable(
  queries,
  corpus.chunks.map((c) => ({
    chunkId: 0,
    docFile: c.docFile,
    docTitle: c.docTitle,
    heading: c.heading,
    headingPath: c.headingPath,
    body: c.text,
    score: 0,
  })),
);
const sectionN = sectionMask.filter(Boolean).length;

// The level every "did it get it right" judgment is made at. Section when the
// ground truth names one, document otherwise — MuPLeR's passages have no
// internal structure to name, and scoring it at a level it does not have would
// report zeros for every strategy alike.
const PRIMARY: Level = DATASET.hasSectionGroundTruth ? "section" : "doc";
const primaryMask = PRIMARY === "section" ? sectionMask : queries.map(() => true);
console.log(`dataset ${DATASET.name} — ${DATASET.label}`);
console.log(
  `${queries.length} queries${PRIMARY === "section" ? ` (${sectionN} scorable at section level)` : ""}, candidateK=${CONFIG.candidateK}, evalK=${CONFIG.evalK}\n`,
);

// ---------------------------------------------------------------- query vectors
const QCACHE = `${DATASET.dir}/query-vectors.json`;
let queryVectors: number[][];
if (existsSync(QCACHE)) {
  const c = JSON.parse(readFileSync(QCACHE, "utf8")) as { n: number; vectors: number[][] };
  if (c.n < corpus.queries.length) throw new Error(`${QCACHE} holds ${c.n} of ${corpus.queries.length} query vectors; delete it.`);
  queryVectors = c.vectors;
} else {
  const r = await embedAll(corpus.queries.map((q) => q.question), { concurrency: 4 });
  writeFileSync(QCACHE, JSON.stringify({ n: corpus.queries.length, vectors: r.vectors }));
  queryVectors = r.vectors;
  console.log(`embedded ${queries.length} queries (${r.promptTokens} tokens)\n`);
}

// ---------------------------------------------------------------- first stage
interface Shortlist {
  queryId: string;
  candidates: Candidate[];
  wallMs: number;
}

const FIRST_STAGES = {
  fts: (q: EvalQuery, _v: number[]) => ftsSearch(q.question),
  vector: (_q: EvalQuery, v: number[]) => vectorSearch(v),
  hybrid: (q: EvalQuery, v: number[]) => hybridSearch(q.question, v),
} as const;
type StageName = keyof typeof FIRST_STAGES;

// The cache is keyed by depth: a k=100 sweep must not read a k=30 shortlist.
const SHORTLIST_CACHE = `${DATASET.dir}/shortlists-k${CONFIG.candidateK}.json`;
let shortlists: Record<StageName, Shortlist[]>;

// Shortlists are always built for the whole query set and cached whole, then
// sliced. Caching only the first `--limit` of them once cost a full benchmark
// run: the smoke test wrote three, and the real run read three.
if (existsSync(SHORTLIST_CACHE) && !process.argv.includes("--refresh")) {
  const cached = JSON.parse(readFileSync(SHORTLIST_CACHE, "utf8")) as Record<StageName, Shortlist[]>;
  const have = cached.hybrid?.length ?? 0;
  if (have < corpus.queries.length) {
    throw new Error(`${SHORTLIST_CACHE} holds ${have} of ${corpus.queries.length} queries. Delete it or pass --refresh.`);
  }
  shortlists = { fts: cached.fts.slice(0, limit), vector: cached.vector.slice(0, limit), hybrid: cached.hybrid.slice(0, limit) };
  console.log("first stage: reusing cached shortlists (--refresh to re-query)\n");
} else {
  const full: Record<StageName, Shortlist[]> = { fts: [], vector: [], hybrid: [] };
  for (const stage of Object.keys(FIRST_STAGES) as StageName[]) {
    const t = Date.now();
    for (let i = 0; i < corpus.queries.length; i++) {
      const r = await FIRST_STAGES[stage](corpus.queries[i]!, queryVectors[i]!);
      full[stage].push({ queryId: corpus.queries[i]!.id, candidates: r.candidates, wallMs: r.wallMs });
    }
    const ms = full[stage].map((s) => s.wallMs);
    console.log(`first stage ${stage.padEnd(7)} ${corpus.queries.length} queries in ${((Date.now() - t) / 1000).toFixed(1)}s (p50 ${percentile(ms, 50)}ms, p95 ${percentile(ms, 95)}ms)`);
  }
  writeFileSync(SHORTLIST_CACHE, JSON.stringify(full));
  shortlists = { fts: full.fts.slice(0, limit), vector: full.vector.slice(0, limit), hybrid: full.hybrid.slice(0, limit) };
  console.log();
}

// ---------------------------------------------------------------- strategies
interface StrategyResult {
  name: string;
  stage: StageName;
  reranker: string;
  /** Reordered lists, aligned with `queries`. */
  ranked: Candidate[][];
  /** The reranker score of each query's top-ranked candidate, for the
   *  abstention analysis. `null` where the strategy produces no score. */
  topScores: (number | null)[];
  /**
   * Every candidate's score, per query, as [chunkId, score] pairs.
   *
   * Kept so that fusing rankers is an offline exercise: once these are on disk,
   * any combination of them — RRF, CombSUM, Borda, a weighted sum — can be
   * searched without issuing a single further API call. The runs that produce
   * them are the expensive part; the search over them should be free.
   */
  scoresByQuery: [number, number][][];
  rerankWallMs: number[];
  inputTokens: number;
  outputTokens: number;
  requests: number;
  failed: number;
  usdCost: number;
}

const results: StrategyResult[] = [];

function passthrough(stage: StageName): StrategyResult {
  return {
    name: stage,
    stage,
    reranker: "none",
    ranked: shortlists[stage].map((s) => s.candidates),
    topScores: shortlists[stage].map(() => null),
    // The first stage's own score is already on the candidates.
    scoresByQuery: shortlists[stage].map((s) => s.candidates.map((c) => [c.chunkId, c.score] as [number, number])),
    rerankWallMs: shortlists[stage].map(() => 0),
    inputTokens: 0,
    outputTokens: 0,
    requests: 0,
    failed: 0,
    usdCost: 0,
  };
}

for (const stage of ["fts", "vector", "hybrid"] as StageName[]) {
  if (wanted(stage)) results.push(passthrough(stage));
}

type Reranker = (question: string, candidates: Candidate[]) => Promise<RerankOutcome>;

async function runReranker(name: string, stage: StageName, reranker: Reranker, costPerMillionInput: number) {
  const t0 = Date.now();
  const ranked: Candidate[][] = [];
  const topScores: (number | null)[] = [];
  const scoresByQuery: [number, number][][] = [];
  const rerankWallMs: number[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let requests = 0;
  let failed = 0;

  for (let i = 0; i < queries.length; i++) {
    const q = queries[i]!;
    const candidates = shortlists[stage][i]!.candidates;
    const t = Date.now();
    const out = await reranker(q.question, candidates);
    rerankWallMs.push(Date.now() - t);
    ranked.push(out.ranked);
    const top = out.ranked[0];
    topScores.push(top ? out.scores.get(top.chunkId) ?? null : null);
    scoresByQuery.push(candidates.map((c) => [c.chunkId, out.scores.get(c.chunkId) ?? Number.NaN] as [number, number]));
    inputTokens += out.inputTokens;
    outputTokens += out.outputTokens;
    requests += out.requests;
    failed += out.failed;
    if ((i + 1) % 20 === 0 || i + 1 === queries.length) {
      process.stdout.write(`\r  ${name} ${i + 1}/${queries.length} (${((Date.now() - t0) / 1000).toFixed(0)}s)   `);
    }
  }
  process.stdout.write("\n");

  results.push({
    name,
    stage,
    reranker: name.split("+")[1] ?? name,
    ranked,
    topScores,
    scoresByQuery,
    rerankWallMs,
    inputTokens,
    outputTokens,
    requests,
    failed,
    usdCost: (inputTokens / 1e6) * costPerMillionInput,
  });
}

const JEV_COST = CONFIG.jevInputCostPerMillion;
const VOYAGE_COST = 0.05; // rerank-2.5 list price per million tokens, 2026-09

if (wanted("hybrid+jev-noul")) await runReranker("hybrid+jev-noul", "hybrid", jevRerankPerCall, JEV_COST);
if (wanted("hybrid+jev-batched")) await runReranker("hybrid+jev-batched", "hybrid", (q, c) => jevRerankBatched(q, c), JEV_COST);
if (wanted("hybrid+jev-score")) await runReranker("hybrid+jev-score", "hybrid", jevRerankScore, JEV_COST);
if (wanted("hybrid+jev-cascade")) await runReranker("hybrid+jev-cascade", "hybrid", (q, c) => jevRerankCascade(q, c), JEV_COST);
if (wanted("vector+jev-noul")) await runReranker("vector+jev-noul", "vector", jevRerankPerCall, JEV_COST);
if (wanted("vector+jev-batched")) await runReranker("vector+jev-batched", "vector", (q, c) => jevRerankBatched(q, c), JEV_COST);
if (wanted("fts+jev-noul")) await runReranker("fts+jev-noul", "fts", jevRerankPerCall, JEV_COST);

// The comparison that decides whether a typed-judgment model is worth having:
// the same question, asked of a general chat model. Two generations, because
// they cannot be asked the same way — see rerank/llm.ts.
if (llmConfigured()) {
  for (const variant of Object.values(LLM_VARIANTS)) {
    const name = `hybrid+${variant.name}`;
    if (wanted(name)) await runReranker(name, "hybrid", llmRerank(variant), variant.inputCostPerMillion);
  }
}

// Open-weight control, scored offline by scripts/laya_score.py. Free to run
// and absent unless that script has been run for this corpus.
for (const kind of ["noul", "score"] as const) {
  const name = `hybrid+laya-${kind}`;
  const table = loadPrecomputed(`laya-scores-${kind}`);
  if (wanted(name) && table) {
    await runReranker(name, "hybrid", precomputedRerank(table, queries.map((q) => q.id)), 0);
  }
}

if (wanted("hybrid+voyage") && voyageConfigured()) {
  await runReranker(
    "hybrid+voyage",
    "hybrid",
    async (question, candidates) => {
      const r = await voyageRerank(question, candidates.map(passageText));
      const scores = new Map<number, number>();
      candidates.forEach((c, i) => scores.set(c.chunkId, r.scores[i]!));
      return {
        ranked: [...candidates].sort((a, b) => scores.get(b.chunkId)! - scores.get(a.chunkId)!),
        scores,
        inputTokens: r.totalTokens,
        outputTokens: 0,
        requests: 1,
        failed: 0,
      };
    },
    VOYAGE_COST,
  );
}

// ---------------------------------------------------------------- scoring
/**
 * A search engine that knows when it has failed is worth more than one that is
 * slightly more accurate, because the failure can be handed to a person instead
 * of being answered confidently from the wrong collective agreement. A Noul is
 * a probability, so it can be thresholded directly; a Score is a rubric level,
 * so its thresholds are on a different scale. Both are reported raw.
 */
function abstentionCurve(
  r: StrategyResult,
  sectionRows: { firstRelevantRank: number | null }[],
  mask: boolean[],
): { threshold: number; answered: number; precisionWhenAnswered: number }[] {
  const scores = r.topScores.map((s, i) => (mask[i] ? s : null));
  if (scores.every((s) => s === null)) return [];
  const max = Math.max(...scores.filter((s): s is number => s !== null));
  // Nouls live in [0,1]; Score rubric levels run to the rubric length.
  const thresholds = max > 1.5 ? [0, 1, 1.5, 2, 2.5, 3] : [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9];
  return thresholds.map((threshold) => {
    const answeredIdx = scores.map((s, i) => (s !== null && s >= threshold ? i : -1)).filter((i) => i >= 0);
    const correct = answeredIdx.filter((i) => sectionRows[i]!.firstRelevantRank === 1).length;
    const evaluable = mask.filter(Boolean).length || 1;
    return {
      threshold,
      answered: answeredIdx.length / evaluable,
      precisionWhenAnswered: answeredIdx.length > 0 ? correct / answeredIdx.length : 0,
    };
  });
}

interface Scored {
  name: string;
  stage: StageName;
  reranker: string;
  doc: Aggregate;
  section: Aggregate;
  ceilingDoc: number;
  ceilingSection: number;
  rerankP50Ms: number;
  rerankP95Ms: number;
  retrieveP50Ms: number;
  retrieveP95Ms: number;
  requests: number;
  failed: number;
  inputTokens: number;
  usdCost: number;
  usdPer1000Queries: number;
  /**
   * Can the score itself tell a failed retrieval from a good one? For each
   * threshold: the share of queries the system would answer, and the share of
   * those answers whose top hit is the right section.
   */
  abstention: { threshold: number; answered: number; precisionWhenAnswered: number }[];
  /** Per-candidate scores, for offline fusion. */
  scoresByQuery: [number, number][][];
  perQuery: { id: string; docRank: number | null; sectionRank: number | null; topScore: number | null }[];
}

const scored: Scored[] = results.map((r) => {
  const levels = (["doc", "section"] as Level[]).map((level) => ({
    level,
    rows: r.ranked.map((list, i) => scoreQuery(list, queries[i]!, level, CONFIG.evalK)),
  }));
  const doc = aggregate(levels.find((l) => l.level === "doc")!.rows);
  const section = aggregate(levels.find((l) => l.level === "section")!.rows.filter((_r, i) => sectionMask[i]));
  const stageWall = shortlists[r.stage].map((s) => s.wallMs);
  const pairs = shortlists[r.stage].map((s, i) => ({ q: queries[i]!, candidates: s.candidates }));
  const sectionPairs = pairs.filter((_p, i) => sectionMask[i]);

  return {
    name: r.name,
    stage: r.stage,
    reranker: r.reranker,
    doc,
    section,
    ceilingDoc: shortlistCeiling(pairs, "doc"),
    ceilingSection: shortlistCeiling(sectionPairs, "section"),
    rerankP50Ms: percentile(r.rerankWallMs, 50),
    rerankP95Ms: percentile(r.rerankWallMs, 95),
    retrieveP50Ms: percentile(stageWall, 50),
    retrieveP95Ms: percentile(stageWall, 95),
    requests: r.requests,
    failed: r.failed,
    inputTokens: r.inputTokens,
    usdCost: r.usdCost,
    usdPer1000Queries: (r.usdCost / (queries.length || 1)) * 1000,
    scoresByQuery: r.scoresByQuery,
    abstention: abstentionCurve(r, levels.find((l) => l.level === PRIMARY)!.rows, primaryMask),
    perQuery: r.ranked.map((list, i) => ({
      id: queries[i]!.id,
      docRank: levels.find((l) => l.level === "doc")!.rows[i]!.firstRelevantRank,
      sectionRank: levels.find((l) => l.level === "section")!.rows[i]!.firstRelevantRank,
      topScore: r.topScores[i] ?? null,
    })),
  };
});

/**
 * `--append` merges this run's strategies into the existing `latest.json`
 * instead of replacing it, so adding one baseline costs one baseline's worth of
 * API calls rather than re-running the whole matrix. The queries and the
 * shortlist depth have to match, or the rows would not be comparable and
 * merging them would quietly produce a table that was never measured together.
 */
const resultsDir = `results/${DATASET.name}`;
mkdirSync(resultsDir, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const outPath = `${resultsDir}/run-k${CONFIG.candidateK}-${stamp}.json`;
// `--tag=<name>` writes to `latest-<name>.json` instead of `latest.json`, so an
// ablation cannot overwrite the run it is being compared against.
const tag = process.argv.find((a) => a.startsWith("--tag="))?.split("=")[1];
const latestPath = `${resultsDir}/latest${CONFIG.candidateK === 30 ? "" : `-k${CONFIG.candidateK}`}${tag ? `-${tag}` : ""}.json`;
let merged = scored;
if (process.argv.includes("--append") && existsSync(latestPath)) {
  const prev = JSON.parse(readFileSync(latestPath, "utf8")) as {
    config: { candidateK: number };
    queries: { id: string }[];
    strategies: Scored[];
  };
  if (prev.config.candidateK !== CONFIG.candidateK || prev.queries.length !== queries.length) {
    throw new Error(
      `--append refused: ${latestPath} holds ${prev.queries.length} queries at k=${prev.config.candidateK}, this run has ${queries.length} at k=${CONFIG.candidateK}.`,
    );
  }
  const fresh = new Set(scored.map((s) => s.name));
  merged = [...prev.strategies.filter((s) => !fresh.has(s.name)), ...scored];
  console.log(`--append: kept ${merged.length - scored.length} earlier strategies, added ${scored.length}`);
}

writeFileSync(
  outPath,
  JSON.stringify(
    {
      ranAt: new Date().toISOString(),
      dataset: DATASET,
      config: CONFIG,
      model: CONFIG.jevModel,
      primaryLevel: PRIMARY,
      sectionEvaluable: sectionN,
      queries: queries.map((q, i) => ({ id: q.id, question: q.question, category: q.category, files: q.files, sections: q.sections, sectionEvaluable: sectionMask[i], primaryEvaluable: primaryMask[i] })),
      strategies: merged,
    },
    null,
    2,
  ),
  "utf8",
);
writeFileSync(latestPath, readFileSync(outPath, "utf8"), "utf8");

// ---------------------------------------------------------------- console table
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const pad = (s: string, n: number) => s.padEnd(n);
console.log(`\nDOCUMENT LEVEL  (is the right document in the top k)`);
console.log(`${pad("strategy", 22)} ${pad("R@1", 7)}${pad("R@3", 7)}${pad("R@5", 7)}${pad("R@10", 7)}${pad("MRR", 7)}${pad("nDCG", 7)}${pad("ceil", 7)}`);
for (const s of merged) {
  console.log(
    `${pad(s.name, 22)} ${pad(pct(s.doc.recallAt[1]!), 7)}${pad(pct(s.doc.recallAt[3]!), 7)}${pad(pct(s.doc.recallAt[5]!), 7)}${pad(pct(s.doc.recallAt[10]!), 7)}${pad(s.doc.mrr.toFixed(3), 7)}${pad(s.doc.ndcg.toFixed(3), 7)}${pad(pct(s.ceilingDoc), 7)}`,
  );
}
if (PRIMARY === "section") {
  console.log(`\nSECTION LEVEL  (is the right paragraph in the top k; ${sectionN}/${queries.length} queries scorable)`);
  console.log(`${pad("strategy", 22)} ${pad("R@1", 7)}${pad("R@3", 7)}${pad("R@5", 7)}${pad("R@10", 7)}${pad("MRR", 7)}${pad("nDCG", 7)}${pad("ceil", 7)}`);
  for (const s of merged) {
    console.log(
      `${pad(s.name, 22)} ${pad(pct(s.section.recallAt[1]!), 7)}${pad(pct(s.section.recallAt[3]!), 7)}${pad(pct(s.section.recallAt[5]!), 7)}${pad(pct(s.section.recallAt[10]!), 7)}${pad(s.section.mrr.toFixed(3), 7)}${pad(s.section.ndcg.toFixed(3), 7)}${pad(pct(s.ceilingSection), 7)}`,
    );
  }
}
console.log(`\nCOST AND LATENCY`);
console.log(`${pad("strategy", 22)} ${pad("retrieve p50", 14)}${pad("rerank p50", 12)}${pad("rerank p95", 12)}${pad("reqs", 8)}${pad("in-tokens", 12)}${pad("$/1k queries", 13)}${pad("failed", 7)}`);
for (const s of merged) {
  console.log(
    `${pad(s.name, 22)} ${pad(`${s.retrieveP50Ms}ms`, 14)}${pad(s.rerankP50Ms ? `${s.rerankP50Ms}ms` : "-", 12)}${pad(s.rerankP95Ms ? `${s.rerankP95Ms}ms` : "-", 12)}${pad(String(s.requests), 8)}${pad(s.inputTokens.toLocaleString("en-US"), 12)}${pad(s.usdCost ? `$${s.usdPer1000Queries.toFixed(2)}` : "-", 13)}${pad(String(s.failed), 7)}`,
  );
}
console.log(`\nwrote ${outPath}`);
await close();
