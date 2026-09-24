/**
 * Embed `data/corpus.json` and load it into Postgres.
 *
 * Idempotent at the level of the whole corpus: the schema is recreated, so a
 * rerun reloads everything. Vectors are cached by content under
 * `data/embeddings.*`, so a failed upload, or a change to which documents are
 * indexed, costs only the chunks that actually changed.
 */
import { readFileSync } from "node:fs";
import { CONFIG } from "../config";
import { embeddingInput, type Chunk, type SourceDoc } from "../corpus";
import { DATASET } from "../dataset-config";
import { close, sql } from "../db";
import { cacheKey, embedAll, readVectorCache, toPgVector, writeVectorCache } from "../embed";

const t0 = Date.now();
const corpus = JSON.parse(readFileSync(`${DATASET.dir}/corpus.json`, "utf8")) as {
  docs: SourceDoc[];
  chunks: Chunk[];
};
const { docs, chunks } = corpus;
console.log(`dataset ${DATASET.name} -> schema ${DATASET.schema}`);
console.log(`corpus: ${docs.length} documents, ${chunks.length} chunks`);

const CACHE = `${DATASET.dir}/embeddings`;
const inputs = chunks.map(embeddingInput);
const cached = await readVectorCache(CACHE, CONFIG.embeddingDims);
const vectors: number[][] = new Array(chunks.length);
const missing: number[] = [];
for (let i = 0; i < inputs.length; i++) {
  const hit = cached.get(inputs[i]!);
  if (hit) vectors[i] = hit;
  else missing.push(i);
}
console.log(`embeddings: ${chunks.length - missing.length} cached, ${missing.length} to compute`);

if (missing.length > 0) {
  let lastLog = 0;
  const r = await embedAll(
    missing.map((i) => inputs[i]!),
    {
      onProgress: (done, total) => {
        if (done - lastLog >= 2000 || done === total) {
          lastLog = done;
          console.log(`  embedding ${done}/${total} (${((done / total) * 100).toFixed(0)}%) ${((Date.now() - t0) / 1000).toFixed(0)}s`);
        }
      },
    },
  );
  missing.forEach((i, j) => (vectors[i] = r.vectors[j]!));
  console.log(`embeddings: ${r.promptTokens} tokens, ~$${((r.promptTokens / 1e6) * 0.13).toFixed(3)} at text-embedding-3-large rates`);

  // Rewrite the cache as the union of what it held and what this run computed,
  // deduplicated: two identical chunks in different documents share a vector.
  const union = new Map<string, number[]>();
  for (let i = 0; i < inputs.length; i++) union.set(cacheKey(inputs[i]!), vectors[i]!);
  await writeVectorCache(CACHE, [...union].map(([key, vector]) => ({ key, vector })), CONFIG.embeddingDims);
}

console.log("applying schema…");
await sql.unsafe(readFileSync("db/schema.sql", "utf8").replaceAll("{{schema}}", DATASET.schema));

await sql`INSERT INTO documents ${sql(docs.map((d) => ({ file: d.file, title: d.title, folder: d.folder, bytes: d.bytes })), "file", "title", "folder", "bytes")}`;
console.log(`documents: ${docs.length} rows`);

const BATCH = 400;
for (let i = 0; i < chunks.length; i += BATCH) {
  const slice = chunks.slice(i, i + BATCH);
  const rows = slice.map((c, j) => ({
    doc_file: c.docFile,
    ordinal: c.ordinal,
    heading: c.heading,
    heading_path: c.headingPath,
    body: c.text,
    embedding: toPgVector(vectors[i + j]!),
  }));
  await sql`INSERT INTO chunks ${sql(rows, "doc_file", "ordinal", "heading", "heading_path", "body", "embedding")}`;
  if ((i / BATCH) % 10 === 0 || i + BATCH >= chunks.length) {
    console.log(`  chunks ${Math.min(i + BATCH, chunks.length)}/${chunks.length} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}

console.log("building HNSW index…");
await sql.unsafe(readFileSync("db/index.sql", "utf8").replaceAll("{{schema}}", DATASET.schema));

const [stats] = await sql`
  SELECT count(*)::int AS chunks,
         count(DISTINCT doc_file)::int AS docs,
         pg_size_pretty(pg_total_relation_size('chunks')) AS size
  FROM chunks`;
console.log(`\ndone in ${((Date.now() - t0) / 1000).toFixed(0)}s — ${stats!.chunks} chunks, ${stats!.docs} docs, ${stats!.size}`);
await close();
