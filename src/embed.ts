import { env } from "./env";
import { CONFIG } from "./config";

/**
 * Azure OpenAI embeddings, `text-embedding-3-large` at 1536 dimensions.
 * Any embedding provider works; this one is here because both corpora are
 * Finnish and it was already available.
 */

export interface EmbedResult {
  vectors: number[][];
  promptTokens: number;
}

const MAX_INPUT_CHARS = 24_000; // ~8k tokens, the model ceiling, cut in code so a
                                // long chunk is truncated loudly rather than by Azure.

export async function embedBatch(texts: string[], dims: number = CONFIG.embeddingDims): Promise<EmbedResult> {
  const url = `https://${env.azureResource}.openai.azure.com/openai/deployments/${env.azureEmbeddingDeployment}/embeddings?api-version=${env.azureEmbeddingApiVersion}`;
  const input = texts.map((t) => (t.length > MAX_INPUT_CHARS ? t.slice(0, MAX_INPUT_CHARS) : t));

  let lastError: unknown;
  for (let attempt = 0; attempt < 5; attempt++) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "api-key": env.azureKey, "content-type": "application/json" },
      body: JSON.stringify({ input, dimensions: dims }),
    });
    if (res.ok) {
      const json = (await res.json()) as {
        data: { index: number; embedding: number[] }[];
        usage: { prompt_tokens: number };
      };
      const vectors = new Array<number[]>(input.length);
      for (const d of json.data) vectors[d.index] = d.embedding;
      return { vectors: vectors as number[][], promptTokens: json.usage.prompt_tokens };
    }
    const body = await res.text();
    lastError = new Error(`Azure embeddings ${res.status}: ${body.slice(0, 300)}`);
    // 429 carries a retry-after; everything else gets plain exponential backoff.
    if (res.status === 429 || res.status >= 500) {
      const retryAfter = Number(res.headers.get("retry-after") ?? 0);
      const waitMs = retryAfter > 0 ? retryAfter * 1000 : Math.min(30_000, 1000 * 2 ** attempt);
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }
    throw lastError;
  }
  throw lastError;
}

/** Runs batches with bounded concurrency and reports progress. */
export async function embedAll(
  texts: string[],
  opts: { batchSize?: number; concurrency?: number; dims?: number; onProgress?: (done: number, total: number) => void } = {},
): Promise<EmbedResult> {
  const batchSize = opts.batchSize ?? CONFIG.embeddingBatchSize;
  const concurrency = opts.concurrency ?? 6;
  const batches: { start: number; texts: string[] }[] = [];
  for (let i = 0; i < texts.length; i += batchSize) batches.push({ start: i, texts: texts.slice(i, i + batchSize) });

  const vectors = new Array<number[]>(texts.length);
  let promptTokens = 0;
  let done = 0;
  let next = 0;

  const worker = async () => {
    while (true) {
      const i = next++;
      const batch = batches[i];
      if (!batch) return;
      const r = await embedBatch(batch.texts, opts.dims);
      for (let j = 0; j < r.vectors.length; j++) vectors[batch.start + j] = r.vectors[j]!;
      promptTokens += r.promptTokens;
      done += batch.texts.length;
      opts.onProgress?.(done, texts.length);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, batches.length) }, worker));
  return { vectors: vectors as number[][], promptTokens };
}

/**
 * pgvector accepts the literal as text, so the precision written here is the
 * size of the upload. A raw `join(",")` emits ~22 characters per component,
 * which is 34 KB per row and 1.2 GB for this corpus. Six decimals is far below
 * the noise floor of a normalised 1536-dimension embedding and cuts that to
 * about a third.
 */
export function toPgVector(v: number[]): string {
  let out = "[";
  for (let i = 0; i < v.length; i++) {
    if (i > 0) out += ",";
    out += v[i]!.toFixed(6);
  }
  return `${out}]`;
}

/**
 * A content-addressed embedding cache.
 *
 * Keyed by the text that was embedded, not by position. Changing the corpus
 * selection — dropping a file, adding a distractor, re-chunking one document —
 * then costs only the chunks that actually changed instead of re-paying for all
 * 35 658. The first version of this keyed by array index, and a one-document
 * fix to the corpus would have meant a full re-embed.
 *
 * Stored as two files: the vectors as flat float32, and their keys as JSON.
 * `JSON.stringify` of the vectors themselves builds a gigabyte-scale string
 * before anything reaches disk, which is a good way to lose a 20-minute run to
 * an allocation failure.
 */
export function cacheKey(text: string): string {
  return Bun.hash(text).toString(36);
}

export interface VectorCache {
  get(text: string): number[] | undefined;
  size: number;
}

export async function readVectorCache(basePath: string, dims: number): Promise<VectorCache> {
  const keysFile = Bun.file(`${basePath}.keys.json`);
  const vecFile = Bun.file(`${basePath}.f32`);
  if (!(await keysFile.exists()) || !(await vecFile.exists())) {
    return { get: () => undefined, size: 0 };
  }
  const keys = (await keysFile.json()) as string[];
  const flat = new Float32Array(await vecFile.arrayBuffer());
  if (flat.length !== keys.length * dims) {
    throw new Error(`${basePath} is inconsistent: ${keys.length} keys, ${flat.length / dims} vectors. Delete both files to re-embed.`);
  }
  const index = new Map<string, number>();
  keys.forEach((k, i) => index.set(k, i));
  return {
    size: keys.length,
    get(text) {
      const i = index.get(cacheKey(text));
      return i === undefined ? undefined : Array.from(flat.subarray(i * dims, (i + 1) * dims));
    },
  };
}

export async function writeVectorCache(
  basePath: string,
  entries: { key: string; vector: number[] }[],
  dims: number,
): Promise<void> {
  const flat = new Float32Array(entries.length * dims);
  for (let i = 0; i < entries.length; i++) flat.set(entries[i]!.vector, i * dims);
  await Bun.write(`${basePath}.f32`, flat.buffer as ArrayBuffer);
  await Bun.write(`${basePath}.keys.json`, JSON.stringify(entries.map((e) => e.key)));
}
