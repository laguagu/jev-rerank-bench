# jev-rerank-bench — conventions

Read [README.md](README.md) first, then `docs/findings-*.md` for the numbers. This is an experiment, not a product: the
deliverable is a number you can defend, so anything that weakens the comparison
matters more than anything that makes the code prettier.

## Things that will bite

- **Every reranker must receive the identical shortlist.** Shortlists are built
  once per first stage and cached in `<dataset dir>/shortlists-k<depth>.json`. If a
  reranker is allowed to change shortlist membership, the metrics stop being
  comparable and the whole run is worthless.
- **The shortlist cache always covers every query, never `--limit`.** A smoke
  run that cached a partial shortlist would make the next full run read only that.
- **Every model is asked the identical question, and the question lives in one
  file.** `rerank/questions.ts` holds one `RelevanceSpec` per corpus; `jev.ts`
  and `llm.ts` import it, and the Python Laya baseline reads it from
  `bun run dump-question`. Restate the wording anywhere else and the benchmark
  stops comparing models and starts comparing prompts.
- **A question written for one corpus does not transfer.** The private corpus judgment
  requires the passage to come from the document the question names, because
  hundreds of collective agreements repeat each other and only provenance
  separates a right answer from a wrong one. MuPLeR passages carry a bare
  numeric id, so that condition can never be verified — and the same question
  scored *below no reranking at all* there. The mis-specified run is kept at
  `results/mupler/latest-mis-specified-question.json`
  because the contrast is the most useful result in the repository.
- **Fusion is searched offline, never by re-running.** Each strategy stores
  every candidate's score in `scoresByQuery`, so `bun run fuse` can try hundreds
  of combinations for nothing. It splits queries into dev and test by index
  parity, selects on dev and reports test — with this few queries a free search over
  dozens of combinations will always find a spurious winner otherwise.
- **The HNSW index is not used at this corpus size, and that is correct.**
  `bun run check-index` shows the planner choosing a sequential scan for the
  candidate vector query, and forcing the index with `enable_seqscan=off` is
  slower: reading the HNSW graph costs more than scanning the rows. Two
  consequences: the first stage in every result
  here is **exact** nearest-neighbour rather than approximate, so the quality
  numbers are an upper bound rather than an ANN approximation; and
  `CONFIG.hnswEfSearch` had no effect on any of them. Re-check with
  `check-index` before quoting a latency number or tuning `ef_search`.
- **DDL is always schema-qualified; DML relies on `search_path`.** `db/*.sql`
  names every table as `{{schema}}.x`. With `search_path = mupler, public`, an
  unqualified `DROP TABLE IF EXISTS chunks` resolves to `public.chunks` when
  `mupler.chunks` does not exist yet and deletes the private corpus. Cached
  shortlists hide that loss, because runs read them instead of the database.
- **Report the shortlist ceiling with every result.** A reranker cannot retrieve
  what the first stage missed. A "62% Recall@10" means nothing without the "the
  shortlist only contained the answer 71% of the time" next to it.
- **Do not fold `ä`/`ö` with unaccent.** They are distinct letters in Finnish,
  not accents, and folding them merges words this corpus distinguishes. The
  schema comment says so; leave it.
- **`ts_rank_cd` is not BM25.** This project ranks lexical matches with Postgres
  FTS. Say "Postgres FTS" in any write-up, never "BM25" — the TypeSafe cookbook
  this is compared against
  used real BM25 and the two are not interchangeable.
- **Retrieval latency in `results/` includes a round trip to Frankfurt.** It is
  not a measure of pgvector speed. Rerank latency is the number that is actually
  about the model.
- **Query ids repeat in the source CSV.** The same `node` supplies two different
  questions; `dataset.ts` disambiguates with a `.2` suffix. Do not key anything
  on `node`.
- **`data/<ds>/embeddings.*` is a paid artifact.** It is content-addressed, so a
  chunk whose text is unchanged is never re-embedded. Deleting it means paying
  for, and waiting on, a full re-embed.
- **The distractor sample is deterministic but not stable.** `prepare-corpus`
  shuffles the whole document list with a seeded PRNG and takes a prefix, so
  adding or removing *any* source file — even one that was never selected —
  reshuffles which distractors are chosen, and most of the corpus needs
  re-embedding. If the source tree is going to change often, order the pool by a
  hash of the file
  name instead; that keeps every other selection where it was.

## Two datasets

`--dataset=fi-tes` (default) and `--dataset=mupler` select the corpus. They live
in separate Postgres schemas — `public` and `mupler` — set through `search_path`
in the connection's startup options, so every query in `search.ts` names
`chunks` and `documents` unqualified and still reads the right one.

- **fi-tes** is not redistributable. Aggregate numbers derived from it may be
  published; the documents, the source path and the provider name may not.
- **mupler** is `mteb/MuPLeR-retrieval` fi-split, EUPL-1.2, and is the dataset
  any public version of this work reports on. Section-level ground truth does
  not exist there; only document level is scored.

Results are written to `results/<dataset>/`, and `docs/findings-<dataset>.md`
is generated from them by `bun run report`. Everything under `docs/` except the
image is generated above the `<!-- interpretation -->` marker and hand-written
below it; `report` preserves the hand-written half.

## Changing the corpus

`CONFIG.corpusDocs`, the chunk sizes and the seed are all in `src/config.ts`.
Changing any of them invalidates `data/<ds>/embeddings.*`, `data/<ds>/shortlists-*.json`
and every result file — delete all three together, or the run silently mixes
generations.

## Secrets

Keys come from `~/.agents/env/_common.env` into a gitignored `.env`. Never commit
a real value and never put one in `.env.example`.

## The Laya baseline

`scripts/laya_score.py` runs [Laya](https://github.com/NandhaKishorM/laya)
(Apache-2.0, `convaiinnovations/laya`) from local weights in `.venv-laya`. It
answers the same three primitives as Jev, so it is the open-weight control: if
it matches a hosted model on reranking, the capability is not something that has
to be bought.

- Use the `multilingual` checkpoint. Both corpora are Finnish.
- Its latency in `results/` was measured on CPU, so it is not the model's GPU
  latency. `.venv-laya` is machine-local and gitignored; create it on each machine
  before `bun run laya`.
- It writes `<dataset dir>/laya-scores-<kind>.json`, which the benchmark reads
  as a precomputed reranker. Nothing in that path calls a paid API.
