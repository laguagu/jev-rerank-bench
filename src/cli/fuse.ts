/**
 * Offline fusion search.
 *
 * Every strategy in a run stores the score it gave each candidate, so combining
 * rankers costs nothing: no API call, no database query, just arithmetic over
 * `results/<dataset>/latest.json`. That is the point — the runs are expensive,
 * the search over their outputs should be free, and it can therefore be
 * exhaustive.
 *
 * **The split is not optional.** Searching dozens of combinations against 84 or
 * 200 queries will always turn up a winner by chance. Queries are split into
 * dev and test halves by parity of their index, every combination is chosen on
 * dev alone, and the number reported is the one it then scores on test. A
 * method's dev score is a selection statistic, not a result.
 *
 *   bun run fuse                       # default dataset
 *   bun run fuse --dataset=mupler
 */
import { readFileSync } from "node:fs";
import { CONFIG } from "../config";
import { DATASET } from "../dataset-config";
import type { EvalQuery } from "../dataset";
import { aggregate, scoreQuery, type Level } from "../metrics";
import type { Candidate } from "../search";

interface Strategy {
  name: string;
  stage: string;
  scoresByQuery?: [number, number][][];
}
interface Run {
  primaryLevel: Level;
  queries: (EvalQuery & { primaryEvaluable: boolean })[];
  strategies: Strategy[];
}

const run = JSON.parse(readFileSync(`results/${DATASET.name}/latest.json`, "utf8")) as Run;
const shortlists = JSON.parse(readFileSync(`${DATASET.dir}/shortlists-k${CONFIG.candidateK}.json`, "utf8")) as Record<
  string,
  { candidates: Candidate[] }[]
>;

const withScores = run.strategies.filter((s) => s.scoresByQuery && s.scoresByQuery.length > 0);
if (withScores.length < 2) {
  console.error(
    `results/${DATASET.name}/latest.json has ${withScores.length} strategies with stored candidate scores.\n` +
      `Re-run the benchmark with the current code — earlier runs did not store them.`,
  );
  process.exit(1);
}

const LEVEL = run.primaryLevel;
const nQueries = run.queries.length;
const evaluable = run.queries.map((q) => q.primaryEvaluable !== false);

/** Per-query min-max to [0,1]. A ranker that gave every candidate the same
 *  score contributes nothing rather than dividing by zero. */
function normalise(pairs: [number, number][]): Map<number, number> {
  const vals = pairs.map(([, v]) => (Number.isFinite(v) ? v : 0));
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo;
  const out = new Map<number, number>();
  pairs.forEach(([id, v], i) => out.set(id, span > 0 ? ((Number.isFinite(v) ? v : 0) - lo) / span : 0));
  return out;
}

/** Descending rank, 1-based, ties broken by candidate order. */
function ranks(pairs: [number, number][]): Map<number, number> {
  const sorted = [...pairs].sort((a, b) => (Number.isFinite(b[1]) ? b[1] : -Infinity) - (Number.isFinite(a[1]) ? a[1] : -Infinity));
  const out = new Map<number, number>();
  sorted.forEach(([id], i) => out.set(id, i + 1));
  return out;
}

type Combiner = (perRanker: { norm: Map<number, number>; rank: Map<number, number> }[], id: number, n: number) => number;

const METHODS: Record<string, Combiner> = {
  // Cormack et al. Rank-only, so score scales never have to be reconciled.
  rrf: (rs, id) => rs.reduce((a, r) => a + 1 / (CONFIG.rrfK + (r.rank.get(id) ?? 1e6)), 0),
  // Fox & Shaw. Sum of normalised scores.
  combsum: (rs, id) => rs.reduce((a, r) => a + (r.norm.get(id) ?? 0), 0),
  // CombSUM weighted by how many rankers put the candidate above the midpoint;
  // agreement between rankers is what it rewards.
  combmnz: (rs, id) => {
    const sum = rs.reduce((a, r) => a + (r.norm.get(id) ?? 0), 0);
    const hits = rs.filter((r) => (r.norm.get(id) ?? 0) > 0.5).length;
    return sum * Math.max(1, hits);
  },
  // Borda: points for position, insensitive to score magnitude and to outliers.
  borda: (rs, id, n) => rs.reduce((a, r) => a + (n - (r.rank.get(id) ?? n)), 0),
  // Geometric mean. One ranker calling a candidate irrelevant vetoes it.
  geometric: (rs, id) => rs.reduce((a, r) => a * ((r.norm.get(id) ?? 0) + 0.01), 1),
  // The single best ranker's opinion, i.e. an optimistic union.
  max: (rs, id) => Math.max(...rs.map((r) => r.norm.get(id) ?? 0)),
};

interface Fusion {
  label: string;
  rankers: string[];
  method: string;
  weights?: number[];
}

function evaluateFusion(f: Fusion, indices: number[]): { r1: number; mrr: number; ndcg: number } {
  const byName = new Map(withScores.map((s) => [s.name, s]));
  const rows = indices.map((qi) => {
    const perRanker = f.rankers.map((name, ri) => {
      const pairs = byName.get(name)!.scoresByQuery![qi]!;
      const w = f.weights?.[ri] ?? 1;
      const scaled: [number, number][] = pairs.map(([id, v]) => [id, (Number.isFinite(v) ? v : 0) * w]);
      return { norm: normalise(scaled), rank: ranks(pairs) };
    });
    const stage = byName.get(f.rankers[0]!)!.stage;
    const candidates = shortlists[stage]![qi]!.candidates;
    const combine = METHODS[f.method]!;
    const fused = [...candidates].sort(
      (a, b) => combine(perRanker, b.chunkId, candidates.length) - combine(perRanker, a.chunkId, candidates.length),
    );
    return scoreQuery(fused, run.queries[qi]!, LEVEL, CONFIG.evalK);
  });
  const agg = aggregate(rows);
  return { r1: agg.recallAt[1]!, mrr: agg.mrr, ndcg: agg.ndcg };
}

// ---------------------------------------------------------------- candidates
const names = withScores.map((s) => s.name);
const pairs: string[][] = [];
for (let i = 0; i < names.length; i++) {
  for (let j = i + 1; j < names.length; j++) pairs.push([names[i]!, names[j]!]);
}
const triples: string[][] = [];
for (let i = 0; i < names.length; i++) {
  for (let j = i + 1; j < names.length; j++) {
    for (let k = j + 1; k < names.length; k++) triples.push([names[i]!, names[j]!, names[k]!]);
  }
}

const WEIGHT_GRID = [0.25, 0.5, 1, 2];
const fusions: Fusion[] = [];
for (const rankers of [...pairs, ...triples]) {
  for (const method of Object.keys(METHODS)) {
    fusions.push({ label: `${method}(${rankers.join(" + ")})`, rankers, method });
    // Weighted variants only for pairs, and only for the score-scale methods;
    // a weight has no effect on rank-only fusion.
    if (rankers.length === 2 && method !== "rrf" && method !== "borda") {
      for (const w of WEIGHT_GRID) {
        if (w === 1) continue;
        fusions.push({ label: `${method}(${rankers[0]}×${w} + ${rankers[1]})`, rankers, method, weights: [w, 1] });
      }
    }
  }
}

// Single rankers, as the baseline every fusion has to beat.
for (const name of names) fusions.push({ label: name, rankers: [name], method: "combsum" });

// ---------------------------------------------------------------- dev / test
const all = Array.from({ length: nQueries }, (_, i) => i).filter((i) => evaluable[i]);
const dev = all.filter((_, k) => k % 2 === 0);
const test = all.filter((_, k) => k % 2 === 1);

console.log(`dataset ${DATASET.name}, level ${LEVEL}, ${all.length} evaluable queries (${dev.length} dev / ${test.length} test)`);
console.log(`${withScores.length} rankers with stored scores, ${fusions.length} combinations searched\n`);

const scored = fusions.map((f) => ({ f, dev: evaluateFusion(f, dev), test: evaluateFusion(f, test) }));
const singles = scored.filter((s) => s.f.rankers.length === 1);
const bestSingleDev = [...singles].sort((a, b) => b.dev.r1 - a.dev.r1)[0]!;
const bestFusionDev = [...scored].filter((s) => s.f.rankers.length > 1).sort((a, b) => b.dev.r1 - a.dev.r1)[0]!;

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
console.log("TOP 12 BY DEV R@1 (dev is selection, test is the result)");
console.log(`${"combination".padEnd(52)} ${"dev R@1".padEnd(9)}${"test R@1".padEnd(10)}${"test MRR".padEnd(10)}test nDCG`);
for (const s of [...scored].sort((a, b) => b.dev.r1 - a.dev.r1).slice(0, 12)) {
  console.log(
    `${s.f.label.slice(0, 51).padEnd(52)} ${pct(s.dev.r1).padEnd(9)}${pct(s.test.r1).padEnd(10)}${s.test.mrr.toFixed(3).padEnd(10)}${s.test.ndcg.toFixed(3)}`,
  );
}

console.log("\nSINGLE RANKERS");
for (const s of [...singles].sort((a, b) => b.test.r1 - a.test.r1)) {
  console.log(`${s.f.label.padEnd(52)} ${pct(s.dev.r1).padEnd(9)}${pct(s.test.r1).padEnd(10)}${s.test.mrr.toFixed(3)}`);
}

console.log("\nHELD-OUT VERDICT");
console.log(`  best single ranker, chosen on dev : ${bestSingleDev.f.label}`);
console.log(`     test R@1 ${pct(bestSingleDev.test.r1)}, MRR ${bestSingleDev.test.mrr.toFixed(3)}`);
console.log(`  best fusion, chosen on dev        : ${bestFusionDev.f.label}`);
console.log(`     test R@1 ${pct(bestFusionDev.test.r1)}, MRR ${bestFusionDev.test.mrr.toFixed(3)}`);
const delta = bestFusionDev.test.r1 - bestSingleDev.test.r1;
console.log(
  `\n  fusion ${delta > 0 ? "beats" : delta < 0 ? "loses to" : "ties"} the best single ranker by ${(delta * 100).toFixed(1)} points on held-out queries.`,
);
console.log(`  With ${test.length} test queries, one query is worth ${(100 / test.length).toFixed(1)} points — read small gaps as noise.`);
