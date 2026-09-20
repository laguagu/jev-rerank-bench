import { TypeSafeClient, noul, score as scoreQuestion, type Questions } from "@typesafe-ai/sdk";
import { CONFIG } from "../config";
import { env } from "../env";
import { passageText, type Candidate } from "../search";
import { SPEC } from "./questions";

export interface RerankOutcome {
  /** Candidates in the new order. */
  ranked: Candidate[];
  /** The score each candidate received, keyed by chunk id. */
  scores: Map<number, number>;
  inputTokens: number;
  outputTokens: number;
  requests: number;
  /** Candidates the model never scored because a request failed. */
  failed: number;
}

let client: TypeSafeClient | null = null;
function getClient(): TypeSafeClient {
  client ??= new TypeSafeClient({
    apiKey: env.typesafeKey,
    defaultModel: CONFIG.jevModel,
    timeout: 45_000,
    retry: { maxRetries: 4 },
  });
  return client;
}

/**
 * The judgment under test lives in `questions.ts`, one per corpus, and is
 * shared with the LLM baselines by import so the comparison stays a comparison
 * of models rather than of prompts.
 */
export const RELEVANCE_INSTRUCTIONS = SPEC.instructions;
export const RELEVANCE_CRITERIA = SPEC.criteria;
export const GRADED_RUBRIC = SPEC.rubric;

export function passageState(c: Candidate) {
  return SPEC.state(c);
}

export async function runPool<T, R>(items: T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i]!, i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

export function order(candidates: Candidate[], scores: Map<number, number>): Candidate[] {
  // A failed candidate keeps its position behind everything scored rather than
  // being dropped: the shortlist membership must not change between strategies.
  return [...candidates].sort((a, b) => {
    const sa = scores.get(a.chunkId);
    const sb = scores.get(b.chunkId);
    if (sa === undefined && sb === undefined) return 0;
    if (sa === undefined) return 1;
    if (sb === undefined) return -1;
    return sb - sa;
  });
}

/**
 * One request per candidate, the shape the TypeSafe rerank cookbook uses.
 * Each request sees the question and exactly one passage.
 */
export async function jevRerankPerCall(question: string, candidates: Candidate[]): Promise<RerankOutcome> {
  const client = getClient();
  const scores = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let requests = 0;
  let failed = 0;

  await runPool(candidates, CONFIG.jevConcurrency, async (c) => {
    try {
      const res = await client.systemOne({
        state: { question, passage: passageState(c) },
        questions: { answers_the_question: noul(RELEVANCE_INSTRUCTIONS, RELEVANCE_CRITERIA) },
      });
      scores.set(c.chunkId, res.answers.answers_the_question.noul);
      inputTokens += res.usage.input_tokens;
      outputTokens += res.usage.output_tokens;
    } catch {
      failed++;
    }
    requests++;
  });

  return { ranked: order(candidates, scores), scores, inputTokens, outputTokens, requests, failed };
}

/**
 * One request per batch of candidates: they go into `state.passages` and each
 * gets its own Noul pointing at its index.
 *
 * Independent questions over shared state run in parallel inside one request and
 * cannot see one another, so this is the same set of judgments as the per-call
 * variant, but the question text and the shared framing are billed once instead
 * of once per candidate. Whether the shared context also changes the answers is
 * part of what the benchmark measures.
 */
export async function jevRerankBatched(
  question: string,
  candidates: Candidate[],
  batchSize = 15,
): Promise<RerankOutcome> {
  const client = getClient();
  const scores = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let requests = 0;
  let failed = 0;

  const batches: Candidate[][] = [];
  for (let i = 0; i < candidates.length; i += batchSize) batches.push(candidates.slice(i, i + batchSize));

  await runPool(batches, Math.max(2, Math.floor(CONFIG.jevConcurrency / 2)), async (batch) => {
    const questions: Questions = {};
    batch.forEach((_c, i) => {
      questions[`p${i}`] = noul(
        {
          ...RELEVANCE_INSTRUCTIONS,
          judge:
            "Judge only the passage at `passages[" + i + "]`. The other passages are competing " +
            "candidates for the same question; do not judge them and do not let them change this answer.",
        },
        RELEVANCE_CRITERIA,
      );
    });
    try {
      const res = await client.systemOne({
        state: { question, passages: batch.map(passageState) },
        questions,
      });
      batch.forEach((c, i) => {
        const a = res.answers[`p${i}`];
        if (a && a.type === "noul") scores.set(c.chunkId, a.noul);
        else failed++;
      });
      inputTokens += res.usage.input_tokens;
      outputTokens += res.usage.output_tokens;
    } catch {
      failed += batch.length;
    }
    requests++;
  });

  return { ranked: order(candidates, scores), scores, inputTokens, outputTokens, requests, failed };
}

/**
 * One request per candidate using a graded rubric instead of a yes/no.
 *
 * A Noul collapses "right section, wrong agreement" and "unrelated" into the
 * same low value, which costs ordering information among the near misses that
 * make up most of this shortlist. A Score keeps them apart, and its expected
 * value is continuous, so it also breaks ties a parsed integer cannot.
 */
export async function jevRerankScore(question: string, candidates: Candidate[]): Promise<RerankOutcome> {
  const client = getClient();
  const scores = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let requests = 0;
  let failed = 0;

  await runPool(candidates, CONFIG.jevConcurrency, async (c) => {
    try {
      const res = await client.systemOne({
        state: { question, passage: passageState(c) },
        questions: { usefulness: scoreQuestion(RELEVANCE_INSTRUCTIONS, GRADED_RUBRIC) },
      });
      scores.set(c.chunkId, res.answers.usefulness.score);
      inputTokens += res.usage.input_tokens;
      outputTokens += res.usage.output_tokens;
    } catch {
      failed++;
    }
    requests++;
  });

  return { ranked: order(candidates, scores), scores, inputTokens, outputTokens, requests, failed };
}

/**
 * Cheap pass, then precise pass.
 *
 * The batched variant costs two requests for a 30-candidate shortlist but pays
 * for it in a shared context the per-call variant does not have. The cascade
 * uses it as a filter — keep the `keep` best — and spends per-call requests only
 * on those, so the precise judgment is bought for a third of the requests.
 *
 * This is the shape a production search path would actually use, and the reason
 * it is in the benchmark rather than the README.
 */
export async function jevRerankCascade(
  question: string,
  candidates: Candidate[],
  keep = 10,
): Promise<RerankOutcome> {
  const first = await jevRerankBatched(question, candidates);
  const survivors = first.ranked.slice(0, keep);
  const second = await jevRerankPerCall(question, survivors);

  // The tail keeps the order the cheap pass gave it, behind everything the
  // precise pass scored. Shortlist membership is unchanged either way.
  const tail = first.ranked.slice(keep);
  return {
    ranked: [...second.ranked, ...tail],
    scores: second.scores,
    inputTokens: first.inputTokens + second.inputTokens,
    outputTokens: first.outputTokens + second.outputTokens,
    requests: first.requests + second.requests,
    failed: first.failed + second.failed,
  };
}

export { passageText };
