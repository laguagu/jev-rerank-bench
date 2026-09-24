/**
 * The qualitative half of the result.
 *
 * An aggregate says reranking moved the numbers; it does not say what the model
 * understood. This prints the queries a reranker rescued and the ones it broke,
 * with the passage that used to be first, so the win can be inspected rather
 * than trusted.
 */
import { readFileSync } from "node:fs";
import { CONFIG } from "../config";
import { DATASET } from "../dataset-config";
import type { Candidate } from "../search";

interface PerQuery { id: string; docRank: number | null; sectionRank: number | null; topScore: number | null }
interface Strategy { name: string; stage: string; reranker: string; perQuery: PerQuery[] }
interface Run { primaryLevel: "doc" | "section"; queries: { id: string; question: string; files: string[]; sections: string[]; sectionEvaluable: boolean; primaryEvaluable: boolean }[]; strategies: Strategy[] }

const run = JSON.parse(readFileSync(`results/${DATASET.name}/latest.json`, "utf8")) as Run;
const shortlists = JSON.parse(readFileSync(`${DATASET.dir}/shortlists-k${CONFIG.candidateK}.json`, "utf8")) as Record<string, { candidates: Candidate[] }[]>;

const target = process.argv[2] ?? "hybrid+jev-noul";
const strategy = run.strategies.find((s) => s.name === target);
if (!strategy) {
  console.error(`no strategy "${target}". Available: ${run.strategies.map((s) => s.name).join(", ")}`);
  process.exit(1);
}
const baseline = run.strategies.find((s) => s.name === strategy.stage);
if (!baseline) {
  console.error(`no baseline for stage "${strategy.stage}"`);
  process.exit(1);
}

const BOTTOM = 999;
const rows = run.queries.map((q, i) => ({
  i,
  q,
  before: (run.primaryLevel === "doc" ? baseline.perQuery[i]?.docRank : baseline.perQuery[i]?.sectionRank) ?? BOTTOM,
  after: (run.primaryLevel === "doc" ? strategy.perQuery[i]?.docRank : strategy.perQuery[i]?.sectionRank) ?? BOTTOM,
  topScore: strategy.perQuery[i]?.topScore ?? null,
}));

const rescued = rows.filter((r) => r.q.primaryEvaluable && r.after === 1 && r.before > 3).sort((a, b) => b.before - a.before);
const broken = rows.filter((r) => r.q.primaryEvaluable && r.before === 1 && r.after > 1).sort((a, b) => b.after - a.after);
const missed = rows.filter((r) => r.q.primaryEvaluable && r.before === BOTTOM && r.after === BOTTOM);

const show = (title: string, list: typeof rows, limit = 6) => {
  console.log(`\n${"=".repeat(78)}\n${title}  (${list.length})\n${"=".repeat(78)}`);
  for (const r of list.slice(0, limit)) {
    const before = shortlists[strategy.stage]?.[r.i]?.candidates[0];
    console.log(`\n[${r.q.id}] ${r.q.question}`);
    console.log(`  gold        ${r.q.files.join(", ")}  §  ${r.q.sections.join(" | ")}`);
    console.log(`  rank        ${r.before === BOTTOM ? "not in top 30" : r.before} -> ${r.after === BOTTOM ? "not in top 30" : r.after}${r.topScore !== null ? `  (score ${r.topScore.toFixed(3)})` : ""}`);
    if (before) {
      console.log(`  first stage put first:`);
      console.log(`    doc      ${before.docTitle.slice(0, 68)}`);
      console.log(`    section  ${(before.heading || "(no heading)").slice(0, 68)}`);
    }
  }
};

console.log(`strategy ${strategy.name} against ${baseline.name}, ${run.primaryLevel} level`);
show("RESCUED — gold was below rank 3, reranker put it first", rescued);
show("BROKEN — gold was first, reranker demoted it", broken);
console.log(`\nSTILL MISSED — gold never entered the shortlist: ${missed.length} queries`);
console.log(`  ${missed.map((r) => r.q.id).join(", ")}`);
