/** Checks every external dependency before a long run spends money on a
 *  misconfigured deployment name. */
import { TypeSafeClient, noul } from "@typesafe-ai/sdk";
import { CONFIG } from "../config";
import { close, sql } from "../db";
import { embedBatch } from "../embed";
import { env } from "../env";
import { voyageConfigured, voyageRerank } from "../rerank/voyage";

const line = (name: string, ok: boolean, detail: string) =>
  console.log(`${ok ? "ok  " : "FAIL"}  ${name.padEnd(22)} ${detail}`);

// Azure embeddings
try {
  const t = Date.now();
  const r = await embedBatch(["Ikääntyneen työntekijän työajan lyhentäminen", "Kilometrikorvaus 2026"]);
  const dims = r.vectors[0]!.length;
  line("azure embeddings", dims === CONFIG.embeddingDims, `${dims} dims, ${r.promptTokens} tokens, ${Date.now() - t}ms`);
} catch (e) {
  line("azure embeddings", false, String(e).slice(0, 160));
}

// Jev
try {
  const client = new TypeSafeClient({ apiKey: env.typesafeKey, defaultModel: CONFIG.jevModel });
  const t = Date.now();
  const { answers, model, usage } = await client.systemOne({
    state: { question: "Kuinka monta lomapäivää kertyy kuukaudessa?", passage: "## 12 § Vuosiloma\nTyöntekijälle kertyy 2,5 arkipäivää lomaa kultakin täydeltä lomanmääräytymiskuukaudelta." },
    questions: { relevant: noul("Does the passage answer the question?") },
  });
  line("jev", true, `${model}, noul=${answers.relevant.noul.toFixed(2)}, ${usage.input_tokens} in-tokens, ${Date.now() - t}ms`);
} catch (e) {
  line("jev", false, String(e).slice(0, 160));
}

// Cross-encoder baseline (optional)
if (voyageConfigured()) {
  try {
    const t = Date.now();
    const r = await voyageRerank("Vuosiloman kertyminen", [
      "Työntekijälle kertyy 2,5 arkipäivää lomaa kultakin täydeltä lomanmääräytymiskuukaudelta.",
      "Työnantaja maksaa matkakulut kilometrikorvauksena.",
    ]);
    line("voyage rerank", r.scores[0]! > r.scores[1]!, `scores ${r.scores.map((x) => x.toFixed(3)).join(", ")}, ${r.totalTokens} tokens, ${Date.now() - t}ms`);
  } catch (e) {
    line("voyage rerank", false, String(e).slice(0, 200));
  }
} else line("voyage rerank", true, "skipped (no VOYAGE_API_KEY)");

// Postgres
try {
  const t = Date.now();
  const [v] = await sql`SELECT extversion FROM pg_extension WHERE extname = 'vector'`;
  const [c] = await sql`SELECT count(*)::int AS n FROM information_schema.tables WHERE table_name IN ('chunks','documents')`;
  line("postgres", true, `pgvector ${v?.extversion ?? "not installed"}, ${c!.n}/2 tables, ${Date.now() - t}ms`);
  if (c!.n === 2) {
    const [n] = await sql`SELECT count(*)::int AS chunks, count(embedding)::int AS embedded FROM chunks`;
    line("index", true, `${n!.chunks} chunks, ${n!.embedded} embedded`);
  }
} catch (e) {
  line("postgres", false, String(e).slice(0, 160));
}

await close();
