![Forty near-identical stacks of legal documents on a concrete floor, one lit in amber](docs/hero.jpg)

# jev-rerank-bench

Does reranking with a **typed-judgment model** beat a dedicated cross-encoder?

The picture is the problem. A corpus of Finnish collective agreements contains
hundreds of documents that repeat each other section for section, so
"32 § Ikääntyneiden työntekijöiden työajan lyhentäminen" exists, nearly word for
word, in dozens of them. The passage text does not identify the right answer —
the document it belongs to does. A bi-encoder embedding cannot see that
distinction, because it never sees the query and the passage at the same time.

That is what a reranker is for. This repository measures whether
[TypeSafe Jev](https://docs.typesafe.ai) — which returns a calibrated
probability instead of generated text — does it better than the alternatives, on
two Finnish legal corpora, over the same shortlists, with the same question.

## What it measures

| | |
| --- | --- |
| **First stage** | Postgres FTS (`ts_rank_cd`, `finnish`), pgvector HNSW, and RRF of both |
| **Rerankers** | Jev (noul / score / batched / cascade), Voyage `rerank-2.5`, `gpt-4.1-mini` via logprobs, `gpt-5.6-luna` graded, [Laya](https://github.com/NandhaKishorM/laya) open weights |
| **Metrics** | Recall@1/3/5/10, MRR@10, nDCG@10, shortlist ceiling, latency, \$/1000 queries, abstention curve |
| **Corpora** | `mupler` — [MuPLeR-fi](https://huggingface.co/datasets/mteb/MuPLeR-retrieval), 10 000 EU legal passages, 200 queries, EUPL-1.2, reproducible by anyone. `fi-tes` — 160 Finnish collective agreements and statutes, 37 440 chunks, 84 verified questions with section-level ground truth; not redistributable, so the code path is here and the documents are not. |

Every reranker receives the **identical shortlist**, and every model is asked the
**identical question** — the LLM baselines import the wording from
`src/rerank/questions.ts` rather than restating it, and the Python Laya baseline
reads it from `bun run dump-question`. A difference in the metrics can only come
from the model.

## Results

**MuPLeR-fi, 200 queries, document level.** Full tables in
[docs/findings-mupler.md](docs/findings-mupler.md).

| strategy | R@1 | MRR@10 | rerank p50 | \$/1000 queries |
| --- | --- | --- | --- | --- |
| `hybrid` (no reranking) | 73.0% | 0.797 | — | — |
| `hybrid+llm-4.1-mini` | 92.5% | 0.946 | 1819 ms | \$5.09 |
| `hybrid+llm-5.6-luna` | 93.0% | 0.949 | 3259 ms | \$2.76 |
| `hybrid+voyage` | 95.5% | 0.962 | 333 ms | **\$0.51** |
| **`hybrid+jev-batched`** | **97.0%** | **0.970** | **334 ms** | \$0.60 |

97.0% is the **shortlist ceiling**: batched Jev puts the gold passage first on
every query where the first stage retrieved it at all. Nothing better exists at
this depth.

**fi-tes, 84 queries, section level.** Full tables in [docs/findings-fi-tes.md](docs/findings-fi-tes.md).

| strategy | R@1 | MRR@10 | \$/1000 queries |
| --- | --- | --- | --- |
| `hybrid` (no reranking) | 31.9% | 0.411 | — |
| `hybrid+laya-noul` (open weights) | 8.3% | 0.148 | \$0 |
| `hybrid+llm-4.1-mini` | 40.3% | 0.486 | \$7.30 |
| `hybrid+jev-batched` | 56.9% | 0.601 | \$0.90 |
| **`hybrid+voyage`** | **58.3%** | **0.619** | \$0.74 |

**The ranking flips between corpora.** On the harder one — human-written
questions, near-duplicate documents — the dedicated cross-encoder is still ahead
by 2.7 points. Anyone claiming a single winner has measured one corpus.

### The largest effect is not a model

The judgment used on `fi-tes` demands provenance: the passage must come from the
document the question names. MuPLeR passages carry a bare numeric id, so that
condition can never be verified — and a Noul that cannot verify its condition
answers no. Same model, same shortlists, same metrics:

| relevance judgment | MuPLeR `jev-noul` | MuPLeR `jev-batched` | fi-tes `jev-batched` |
| --- | --- | --- | --- |
| written for `fi-tes` | 66.5% | 77.0% | **55.6%** |
| written for the passage task | **95.5%** | **97.0%** | 51.4% |

MuPLeR `hybrid` scores 73.0% without any reranking, so the mis-specified question
made reranking *actively harmful*. Thirty points moved on the wording of one
criterion and the removal of two state fields — larger than every model
difference in this repository combined.

Each corpus is best served by the judgment written for what actually separates a
right answer there. Both directions are measured; neither question is
universally better.

### It knows when it is right

MuPLeR, document level, gated on the top-1 score:

| threshold | `jev-batched` | `voyage` | `llm-4.1-mini` |
| --- | --- | --- | --- |
| 0.5 | 96% answered / 99% correct | 98% / 97% | 90% / 94% |
| 0.8 | 78% / 99% | 56% / 99% | 87% / 95% |
| 0.9 | **54% / 100%** | 11% / 100% | 86% / 95% |

Gated at 0.9, batched Jev answers more than half the queries with no top-1 errors
at all. Voyage reaches the same precision over 11% of them. Neither chat model
exceeds 95% at any threshold — `gpt-4.1-mini`'s logprobs saturate at 0.000 and
1.000, and **GPT-5.x rejects `logprobs` outright**, so the newer and cheaper model
cannot produce a usable confidence at all.

For a system that must hand uncertain cases to a person, coverage at a fixed
precision is the number that decides the staffing. This is the one result that
holds on both corpora.

### Batching is free quality

One request per 15 candidates instead of one per candidate:

| MuPLeR | requests | rerank p50 | \$/1000 | R@1 |
| --- | --- | --- | --- | --- |
| `jev-noul` (per call) | 6000 | 892 ms | \$0.93 | 95.5% |
| `jev-batched` | 400 | **334 ms** | **\$0.60** | **97.0%** |

Independent questions over shared state are judged in parallel inside one
request, so this is the same set of judgments — the framing is simply billed once
instead of thirty times. TypeSafe's cookbook measures 12x on cost for the *one
document, many questions* shape; this is the other shape, many passages and one
question each, where the passages dominate the tokens and the saving lands on
latency instead.

### Three things that did not work

- **Fusing rankers.** `bun run fuse` searches 245 combinations — RRF, CombSUM,
  CombMNZ, Borda, geometric mean, max, weighted variants — offline and for free,
  selecting on half the queries and reporting the other half. Nothing beats the
  best single ranker on either corpus. The reason is measurable: `jev-batched` and
  `voyage` share 27 of 31 failures (Jaccard 0.79). They are wrong in the same
  places, so there is nothing to combine.
- **A deeper shortlist.** `--candidates=100` lifts the ceiling from 70.8% to 84.7%
  and R@10 by seven points, but R@1 by one to three, at 3.3x the cost. Depth hands
  the reranker more than it can use.
- **Open weights.** Laya answers the same three primitives from local weights and
  was asked the identical question. It scores 8.3% R@1 — below not reranking at
  all — with its scores bunched at the top of the range.

## If you are building search with this

1. **Write the judgment for what separates a right answer in your corpus**, and
   check the condition is verifiable from the state you send. Ours was not, once,
   and cost thirty points. Send only the fields the judgment uses.
2. **Batch.** One request per 10–15 candidates. Cheaper, ~2.7x faster, not worse.
3. **Measure the lexical arm before fusing it.** RRF has no opinion about whether
   a list deserves to be in the fusion: on `fi-tes`, `vector` alone scores 41.7%
   and `hybrid` 31.9%, because the FTS side scores 4.2%. On MuPLeR, where FTS
   reaches 33.5%, fusion helps. One query tells you which case you are in.
4. **Use the probability as a gate, not only an ordering.** That is the capability
   the alternatives lack, and it is worth more than the last two points of
   accuracy.
5. **Report the shortlist ceiling beside every recall number.** A reranker cannot
   retrieve what the first stage missed; on `fi-tes`, 21 of 72 queries never had
   the answer in the top 30.

## Running it

```bash
bun install
cp .env.example .env          # TypeSafe, Azure OpenAI, Postgres; Voyage optional
bun run probe                 # checks every service before spending anything
bun run prepare-mupler        # parquet -> data/mupler/corpus.json
bun run ingest --dataset=mupler
bun run bench  --dataset=mupler
bun run report --dataset=mupler
```

`bun run fuse --dataset=mupler` searches ranker combinations offline.
`bun run examples` prints the queries a reranker rescued and the ones it broke.
`bun run bench --limit=5` is a smoke test. Shortlists, embeddings and query
vectors are cached under `data/`, so reruns do not re-pay.

The MuPLeR parquet files come from the Hugging Face dataset:

```bash
curl -L -o data/mupler/fi-corpus.parquet \
  https://huggingface.co/datasets/mteb/MuPLeR-retrieval/resolve/main/fi-corpus/test-00000-of-00001.parquet
# likewise fi-queries and fi-qrels
```

Any Postgres with pgvector works; `docker-compose.yml` brings up a local one. The
runs in `results/` used Neon in `eu-central-1`, so their retrieval latencies
include a round trip from Finland to Frankfurt and are not a measure of pgvector.

`bun run check-index` prints whether the vector query actually uses the HNSW
index. On this corpus it does not, and the planner is right to decline it — a
sequential scan of 37 440 rows takes 356 ms against 1417 ms for a forced index
scan, because the 283 MB graph does not fit comfortably on a 1 CU compute. **The
first stage in every result here is therefore exact nearest-neighbour, not
approximate**, which makes the retrieval quality an upper bound rather than an
ANN approximation. Check this before quoting any latency number from `results/`.

**Cost.** Every number here was produced for about **\$16** in total — roughly
\$1.40 of embeddings and the rest reranking, across two corpora, twelve
strategies, two shortlist depths and several ablations.

## What is left

- **A real BM25 first stage.** `ts_rank_cd` is not BM25, and Neon has no
  `pg_search`. The claim that a weak lexical arm poisons RRF is about Postgres FTS
  specifically; ParadeDB would very likely change the `fi-tes` numbers.
- **Laya on a GPU, and batched.** 8.3% is an accuracy result and stands, but
  768 ms per decision is a CPU artefact — the project reports 39.5 ms on a T4.
  Whether its scores separate better when several passages share one state is
  untested.
- **Document-level routing.** The `fi-tes` failure mode is "right section, wrong
  agreement". Choosing the agreement first and searching inside it attacks that
  directly, and no reranker over a flat shortlist can.
- **Run-to-run variance.** Two identical `jev-batched` runs differed by about one
  query. Every number here is a single run; none of the sub-two-point gaps should
  be read as real.
- **Real user questions.** MuPLeR's queries were generated from their gold
  passage, which rewards paraphrase matching. `fi-tes` questions are human-written
  and the scores are half as high. That gap is the honest measure of how much a
  synthetic benchmark flatters a reranker.

## Licence

MIT. MuPLeR-fi is EUPL-1.2 and is downloaded, not vendored. The `fi-tes` corpus
is not included.
