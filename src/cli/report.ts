/**
 * Turns `results/<dataset>/latest.json` into `docs/findings-<dataset>.md`.
 *
 * The tables are generated rather than written by hand so a rerun cannot leave a
 * stale number in the prose. Anything interpretive belongs under the tables, in
 * the section this script leaves alone.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { DATASET } from "../dataset-config";

interface Aggregate {
  n: number;
  recallAt: Record<string, number>;
  mrr: number;
  ndcg: number;
}
interface Strategy {
  name: string;
  stage: string;
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
  perQuery: { id: string; docRank: number | null; sectionRank: number | null }[];
}
interface Run {
  ranAt: string;
  primaryLevel: "doc" | "section";
  model: string;
  config: Record<string, unknown>;
  queries: { id: string; question: string; category: string }[];
  strategies: Strategy[];
}

const FINDINGS = `docs/findings-${DATASET.name}.md`;
const run = JSON.parse(readFileSync(`results/${DATASET.name}/latest.json`, "utf8")) as Run;
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const n = run.strategies[0]?.doc.n ?? 0;

function levelTable(level: "doc" | "section"): string {
  const rows = run.strategies.map((s) => {
    const a = level === "doc" ? s.doc : s.section;
    const ceil = level === "doc" ? s.ceilingDoc : s.ceilingSection;
    return `| \`${s.name}\` | ${pct(a.recallAt["1"]!)} | ${pct(a.recallAt["3"]!)} | ${pct(a.recallAt["5"]!)} | ${pct(a.recallAt["10"]!)} | ${a.mrr.toFixed(3)} | ${a.ndcg.toFixed(3)} | ${pct(ceil)} |`;
  });
  return [
    "| strategy | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | shortlist ceiling |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ...rows,
  ].join("\n");
}

function costTable(): string {
  const rows = run.strategies.map(
    (s) =>
      `| \`${s.name}\` | ${s.retrieveP50Ms} ms | ${s.rerankP50Ms || "—"}${s.rerankP50Ms ? " ms" : ""} | ${s.rerankP95Ms || "—"}${s.rerankP95Ms ? " ms" : ""} | ${s.requests || "—"} | ${s.inputTokens ? s.inputTokens.toLocaleString("en-US") : "—"} | ${s.usdCost ? `$${s.usdPer1000Queries.toFixed(2)}` : "—"} | ${s.failed} |`,
  );
  return [
    "| strategy | retrieve p50 | rerank p50 | rerank p95 | requests | input tokens | $ / 1000 queries | failed |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ...rows,
  ].join("\n");
}

/** Which queries each strategy moved, relative to its own first stage. */
function movementTable(): string {
  const byStage = new Map(run.strategies.filter((s) => s.reranker === "none").map((s) => [s.stage, s]));
  const rows: string[] = [];
  for (const s of run.strategies) {
    if (s.reranker === "none") continue;
    const base = byStage.get(s.stage);
    if (!base) continue;
    let better = 0;
    let worse = 0;
    let unchanged = 0;
    const BOTTOM = 999;
    for (let i = 0; i < s.perQuery.length; i++) {
      const key = run.primaryLevel === "doc" ? "docRank" : "sectionRank";
      const a = base.perQuery[i]?.[key] ?? BOTTOM;
      const b = s.perQuery[i]?.[key] ?? BOTTOM;
      if (b < a) better++;
      else if (b > a) worse++;
      else unchanged++;
    }
    rows.push(`| \`${s.name}\` | vs \`${s.stage}\` | ${better} | ${worse} | ${unchanged} |`);
  }
  return ["| strategy | baseline | moved up | moved down | unchanged |", "| --- | --- | --- | --- | --- |", ...rows].join("\n");
}

const header = `# Findings

Generated from \`results/${DATASET.name}/latest.json\` by \`bun run report --dataset=${DATASET.name}\`.
Do not edit the tables by hand — rerun the script.

- **Corpus:** ${DATASET.label}

- **Run:** ${run.ranAt}
- **Model:** ${run.model}
- **Queries:** ${n}
- **Shortlist depth:** ${String(run.config.candidateK)} candidates, metrics at ${String(run.config.evalK)}

## Document level — is the right document in the top k?

${levelTable("doc")}

${run.primaryLevel === "section" ? `## Section level — is the right paragraph in the top k?

${levelTable("section")}` : "_This corpus has no section-level ground truth; only the document level is scored._"}

## Cost and latency

${costTable()}

Retrieval latency is a round trip to a hosted Postgres in \`eu-central-1\`, not a
measure of pgvector itself. Rerank latency is the whole shortlist: the per-call
strategies issue ${String(run.config.candidateK)} requests at a concurrency of
${String(run.config.jevConcurrency)}, the batched strategy issues two.

## Movement against the first stage

How many queries each reranker moved, at ${run.primaryLevel} level. A query the first stage
never retrieved counts as unchanged when the reranker also fails to surface it.

${movementTable()}
`;

const MARKER = "<!-- interpretation -->";
const existing = existsSync(FINDINGS) ? readFileSync(FINDINGS, "utf8") : "";
const kept = existing.includes(MARKER) ? existing.slice(existing.indexOf(MARKER)) : `${MARKER}\n\n## Reading these numbers\n\n_Written by hand; the tables above are generated._\n`;

writeFileSync(FINDINGS, `${header}\n${kept}`, "utf8");
console.log(`wrote ${FINDINGS} (${n} queries, ${run.strategies.length} strategies)`);
