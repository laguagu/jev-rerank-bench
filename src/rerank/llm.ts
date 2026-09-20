import { CONFIG } from "../config";
import { env, optional } from "../env";
import type { Candidate } from "../search";
import {
  GRADED_RUBRIC,
  order,
  passageState,
  runPool,
  RELEVANCE_CRITERIA,
  RELEVANCE_INSTRUCTIONS,
  type RerankOutcome,
} from "./jev";

/**
 * The baseline that actually matters: doing the same judgment with a general
 * chat model.
 *
 * Jev's claim is that a typed judgment is cheaper and faster than prompting a
 * chat model and parsing the answer. That is only testable if the chat model is
 * asked *the same question*, so the instructions, criteria and rubric are
 * imported from the Jev reranker rather than rewritten. What differs is the
 * model and the transport.
 *
 * Two variants, because the two model generations on this Azure resource cannot
 * be asked the same way:
 *
 * - **logprob** (`gpt-4.1-mini`): one token, "yes" or "no", with the answer read
 *   from `logprobs` as P(yes). This is the closest analogue to a Noul.
 * - **graded** (`gpt-5.6-luna`): an integer on the same four-level rubric
 *   `jev-score` uses. GPT-5.x rejects `logprobs` outright — "Unsupported
 *   parameter: 'logprobs' is not supported with this model" — so the newer,
 *   cheaper model can be compared on ranking quality but cannot produce a
 *   calibrated probability at all. That absence is a result, not a limitation
 *   of the harness.
 */
export type LlmMode = "logprob" | "graded";

export interface LlmVariant {
  name: string;
  deployment: string;
  mode: LlmMode;
  apiVersion: string;
  /** USD per million input tokens. Output is at most 16 tokens and ignored. */
  inputCostPerMillion: number;
  /** GPT-5.x only: "none" keeps reasoning tokens at zero. */
  reasoningEffort?: "none" | "low" | "medium" | "high";
}

export const LLM_VARIANTS: Record<string, LlmVariant> = {
  "llm-4.1-mini": {
    name: "llm-4.1-mini",
    deployment: "gpt-4.1-mini",
    mode: "logprob",
    apiVersion: "2024-10-21",
    inputCostPerMillion: 0.4,
  },
  "llm-5.6-luna": {
    name: "llm-5.6-luna",
    deployment: "gpt-5.6-luna",
    mode: "graded",
    apiVersion: "2024-12-01-preview",
    // $0.20/M after the 2026-07-30 cut; half of gpt-4.1-mini and still ~5x Jev.
    inputCostPerMillion: 0.2,
    reasoningEffort: "none",
  },
};

export function llmConfigured(): boolean {
  return optional("AZURE_API_KEY") !== undefined;
}

const YES_NO_SYSTEM = [
  RELEVANCE_INSTRUCTIONS.task,
  RELEVANCE_INSTRUCTIONS.corpus,
  "",
  `Answer yes when: ${RELEVANCE_CRITERIA.true}`,
  `Answer no when: ${RELEVANCE_CRITERIA.false}`,
  "",
  'Reply with exactly one word, "yes" or "no". No punctuation, no explanation.',
].join("\n");

const GRADED_SYSTEM = [
  RELEVANCE_INSTRUCTIONS.task,
  RELEVANCE_INSTRUCTIONS.corpus,
  "",
  "Rate the passage on this scale:",
  ...GRADED_RUBRIC.map((level, i) => `${i} = ${level}`),
  "",
  `Reply with exactly one digit, 0 to ${GRADED_RUBRIC.length - 1}. No punctuation, no explanation.`,
].join("\n");

interface ChatResponse {
  choices: {
    logprobs?: { content?: { token: string; top_logprobs: { token: string; logprob: number }[] }[] };
    message: { content: string | null };
  }[];
  usage: { prompt_tokens: number; completion_tokens: number };
}

function userMessage(question: string, c: Candidate): string {
  const state = passageState(c);
  return `Question: ${question}\n\nPassage:\ndocument: ${state.document}\nsection: ${state.section}\ntext: ${state.text}`;
}

async function scoreOne(
  variant: LlmVariant,
  question: string,
  c: Candidate,
): Promise<{ score: number; inputTokens: number; outputTokens: number }> {
  const url = `https://${env.azureResource}.openai.azure.com/openai/deployments/${variant.deployment}/chat/completions?api-version=${variant.apiVersion}`;
  const messages = [
    { role: "system", content: variant.mode === "graded" ? GRADED_SYSTEM : YES_NO_SYSTEM },
    { role: "user", content: userMessage(question, c) },
  ];

  // GPT-5.x rejects `max_tokens` ("use 'max_completion_tokens' instead") and
  // needs headroom even at zero reasoning effort; GPT-4.x rejects the new name.
  const body =
    variant.mode === "graded"
      ? { messages, max_completion_tokens: 16, reasoning_effort: variant.reasoningEffort ?? "none" }
      : { messages, max_tokens: 1, temperature: 0, logprobs: true, top_logprobs: 8 };

  for (let attempt = 0; attempt < 4; attempt++) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "api-key": env.azureKey, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      const json = (await res.json()) as ChatResponse;
      const usage = { inputTokens: json.usage.prompt_tokens, outputTokens: json.usage.completion_tokens };
      const text = (json.choices[0]?.message.content ?? "").trim();

      if (variant.mode === "graded") {
        const m = /\d/.exec(text);
        // An unparseable answer scores below every rubric level rather than
        // being dropped, so shortlist membership is unchanged.
        return { score: m ? Number(m[0]) : -1, ...usage };
      }

      // Sum the mass on any casing or whitespace variant of yes and of no, then
      // normalise. Reading only the single top token would collapse every
      // candidate the model is unsure about into the same value.
      const top = json.choices[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
      let yes = 0;
      let no = 0;
      for (const t of top) {
        const w = t.token.trim().toLowerCase();
        if (w.startsWith("yes")) yes += Math.exp(t.logprob);
        else if (w.startsWith("no")) no += Math.exp(t.logprob);
      }
      const total = yes + no;
      const score = total > 0 ? yes / total : text.toLowerCase().startsWith("yes") ? 1 : 0;
      return { score, ...usage };
    }
    if (res.status === 429 || res.status >= 500) {
      const retryAfter = Number(res.headers.get("retry-after") ?? 0);
      await new Promise((r) => setTimeout(r, retryAfter > 0 ? retryAfter * 1000 : Math.min(20_000, 1000 * 2 ** attempt)));
      continue;
    }
    throw new Error(`azure chat ${variant.deployment} ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  throw new Error(`azure chat ${variant.deployment}: retries exhausted`);
}

export function llmRerank(variant: LlmVariant) {
  return async (question: string, candidates: Candidate[]): Promise<RerankOutcome> => {
    const scores = new Map<number, number>();
    let inputTokens = 0;
    let outputTokens = 0;
    let requests = 0;
    let failed = 0;

    await runPool(candidates, CONFIG.jevConcurrency, async (c) => {
      try {
        const r = await scoreOne(variant, question, c);
        scores.set(c.chunkId, r.score);
        inputTokens += r.inputTokens;
        outputTokens += r.outputTokens;
      } catch {
        failed++;
      }
      requests++;
    });

    return { ranked: order(candidates, scores), scores, inputTokens, outputTokens, requests, failed };
  };
}
