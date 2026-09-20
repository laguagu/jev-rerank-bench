/**
 * Resolve the ground truth against the source corpus, choose the indexed
 * document set, chunk it, and write `data/corpus.json`.
 *
 * Separate from ingest so the corpus can be inspected before any money is spent
 * on embeddings.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { CONFIG } from "../config";
import { chunkDocument, listDocuments, mulberry32, readDoc, type Chunk, type SourceDoc } from "../corpus";
import { loadGroundTruth, type EvalQuery } from "../dataset";
import { env } from "../env";

const docs = listDocuments(env.corpusSourceDir);
const byFile = new Map(docs.map((d) => [d.file, d]));
const allQueries = loadGroundTruth(env.groundTruthCsv);

const resolved: EvalQuery[] = [];
const dropped: { id: string; missing: string[] }[] = [];
for (const q of allQueries) {
  const missing = q.files.filter((f) => !byFile.has(f));
  if (missing.length > 0 || q.files.length === 0) dropped.push({ id: q.id, missing });
  else resolved.push(q);
}

const goldFiles = new Set(resolved.flatMap((q) => q.files));
const rng = mulberry32(CONFIG.seed);

// Distractors are drawn from the same folders as the gold documents, so the
// index stays dominated by near-identical collective agreements instead of
// being padded with easy out-of-domain text.
const goldFolders = new Set([...goldFiles].map((f) => byFile.get(f)!.folder));
const pool = docs
  .filter((d) => !goldFiles.has(d.file) && goldFolders.has(d.folder))
  .map((d) => ({ d, r: rng() }))
  .sort((a, b) => a.r - b.r)
  .map((x) => x.d);

const selected: SourceDoc[] = [...goldFiles].map((f) => byFile.get(f)!);
for (const d of pool) {
  if (selected.length >= CONFIG.corpusDocs) break;
  selected.push(d);
}

const chunks: Chunk[] = [];
for (const doc of selected) chunks.push(...chunkDocument(doc, readDoc(doc)));

mkdirSync("data", { recursive: true });
writeFileSync(
  "data/corpus.json",
  JSON.stringify({ config: CONFIG, docs: selected, chunks, queries: resolved }),
  "utf8",
);
writeFileSync("data/dropped-queries.json", JSON.stringify(dropped, null, 2), "utf8");

const chars = chunks.reduce((n, c) => n + c.text.length, 0);
const perDoc = new Map<string, number>();
for (const c of chunks) perDoc.set(c.docFile, (perDoc.get(c.docFile) ?? 0) + 1);
const counts = [...perDoc.values()].sort((a, b) => a - b);

console.log(`source corpus      ${docs.length} documents`);
console.log(`ground truth       ${allQueries.length} rows -> ${resolved.length} usable, ${dropped.length} dropped`);
console.log(`indexed documents  ${selected.length} (${goldFiles.size} gold + ${selected.length - goldFiles.size} distractors)`);
console.log(`chunks             ${chunks.length} (${Math.round(chars / chunks.length)} chars avg, ~${Math.round(chars / 4 / 1000)}k tokens total)`);
console.log(`chunks per doc     min ${counts[0]}, median ${counts[Math.floor(counts.length / 2)]}, max ${counts[counts.length - 1]}`);
console.log(`\nwrote data/corpus.json`);
if (dropped.length > 0) {
  console.log(`dropped (file not in corpus): ${dropped.slice(0, 8).map((d) => d.id).join(", ")}${dropped.length > 8 ? ", …" : ""}`);
}
