/**
 * Build `data/mupler/corpus.json` from the MuPLeR Finnish split.
 *
 * MuPLeR (Multilingual Parallel Legal Retrieval, EUPL-1.2) is the public
 * counterpart to the private corpus: EU legal text, Finnish, 10 000 passages and
 * 200 queries with one gold passage each, in BEIR format. The private corpus corpus is
 * employer data and cannot ship with this repository; this one can, so the
 * published numbers are reproducible by anyone.
 *
 * Its documents are already passage-sized, so one document is one chunk and
 * there is no chunking to argue about. Section-level ground truth does not
 * exist here, so only the document level is scored.
 *
 *   curl -L -o data/mupler/fi-corpus.parquet \
 *     https://huggingface.co/datasets/mteb/MuPLeR-retrieval/resolve/main/fi-corpus/test-00000-of-00001.parquet
 *   (likewise fi-queries and fi-qrels)
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parquetReadObjects } from "hyparquet";
import { CONFIG } from "../config";
import type { Chunk, SourceDoc } from "../corpus";
import type { EvalQuery } from "../dataset";

const read = async <T>(path: string): Promise<T[]> => {
  const buf = readFileSync(path);
  const file = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  return (await parquetReadObjects({ file })) as T[];
};

const DIR = "data/mupler";
const corpusRows = await read<{ id: string; text: string; title: string }>(`${DIR}/fi-corpus.parquet`);
const queryRows = await read<{ id: string; text: string }>(`${DIR}/fi-queries.parquet`);
const qrelRows = await read<{ "query-id": string; "corpus-id": string; score: number }>(`${DIR}/fi-qrels.parquet`);

// A qrel score of 0 is an explicit non-match, not a weak match.
const goldByQuery = new Map<string, string[]>();
for (const r of qrelRows) {
  if (Number(r.score) <= 0) continue;
  const list = goldByQuery.get(r["query-id"]);
  if (list) list.push(r["corpus-id"]);
  else goldByQuery.set(r["query-id"], [r["corpus-id"]]);
}

const docs: SourceDoc[] = corpusRows.map((r) => ({
  file: r.id,
  folder: "mupler-fi",
  title: r.title?.trim() || r.id,
  path: `${DIR}/fi-corpus.parquet#${r.id}`,
  bytes: (r.text ?? "").length,
}));

const chunks: Chunk[] = corpusRows.map((r) => ({
  docFile: r.id,
  docTitle: r.title?.trim() || r.id,
  folder: "mupler-fi",
  ordinal: 0,
  heading: r.title?.trim() ?? "",
  headingPath: r.title?.trim() ?? "",
  text: r.text ?? "",
}));

const queries: EvalQuery[] = queryRows
  .filter((q) => (goldByQuery.get(q.id) ?? []).length > 0)
  .map((q) => ({
    id: q.id,
    node: q.id,
    question: q.text,
    files: goldByQuery.get(q.id)!,
    sections: [],
    category: "MUPLER",
    hopType: "single",
    answer: "",
  }));

mkdirSync(DIR, { recursive: true });
writeFileSync(`${DIR}/corpus.json`, JSON.stringify({ config: CONFIG, docs, chunks, queries }), "utf8");

const chars = chunks.reduce((n, c) => n + c.text.length, 0);
const goldCounts = queries.map((q) => q.files.length);
console.log(`documents        ${docs.length}`);
console.log(`chunks           ${chunks.length} (${Math.round(chars / chunks.length)} chars avg, ~${Math.round(chars / 4 / 1000)}k tokens total)`);
console.log(`queries          ${queries.length} of ${queryRows.length} (rest have no positive qrel)`);
console.log(`gold per query   min ${Math.min(...goldCounts)}, max ${Math.max(...goldCounts)}`);
console.log(`\nwrote ${DIR}/corpus.json`);
