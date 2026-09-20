import { optional } from "../env";

/**
 * Cross-encoder baseline. Voyage `rerank-2.5` is multilingual and reads the
 * query and the passage together, which is the thing a bi-encoder embedding
 * cannot do — the same property Jev is being tested for.
 *
 * The Azure-hosted Cohere reranker this workspace used to have was the first
 * choice; its serverless deployment now answers 404 on every route, so the
 * comparison runs against Voyage instead.
 */
const MODEL = process.env.VOYAGE_RERANK_MODEL ?? "rerank-2.5";
const ENDPOINT = "https://api.voyageai.com/v1/rerank";

export interface VoyageRerankResult {
  /** One relevance score per document, in input order. */
  scores: number[];
  totalTokens: number;
}

export function voyageConfigured(): boolean {
  return optional("VOYAGE_API_KEY") !== undefined;
}

export async function voyageRerank(query: string, documents: string[]): Promise<VoyageRerankResult> {
  const key = optional("VOYAGE_API_KEY");
  if (!key) throw new Error("VOYAGE_API_KEY is not set");

  // The model truncates at 8k tokens per document; cut in code so a long chunk
  // is shortened the same way for every strategy.
  const docs = documents.map((d) => (d.length > 12_000 ? d.slice(0, 12_000) : d));

  for (let attempt = 0; attempt < 4; attempt++) {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify({ model: MODEL, query, documents: docs }),
    });
    if (res.ok) {
      const json = (await res.json()) as {
        data: { index: number; relevance_score: number }[];
        usage: { total_tokens: number };
      };
      const scores = new Array<number>(docs.length).fill(0);
      for (const r of json.data) scores[r.index] = r.relevance_score;
      return { scores, totalTokens: json.usage.total_tokens };
    }
    if (res.status === 429 || res.status >= 500) {
      await new Promise((r) => setTimeout(r, Math.min(20_000, 1000 * 2 ** attempt)));
      continue;
    }
    throw new Error(`voyage rerank ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  throw new Error("voyage rerank: retries exhausted");
}
