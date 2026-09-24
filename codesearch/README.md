# Code search: jegrep against an index

Can Jev search a code base without an index, the way
[jegrep](https://github.com/can1357/jegrep) does, and how does that compare with
an embedding index, with and without a Jev rerank?

Self-contained sub-project of `jev-search-lab`; nothing here touches the rerank
benchmark in the parent folder.

## Setup

- **Queries and labels:** jegrep's own benchmark, 10 Postgres and 10 CPython
  queries with annotated files and line regions (`suites/`, MIT, copied from
  jegrep at `e6d5b84`). Each query was also translated into Finnish by hand
  (`suites/fi-queries.json`), because the code is English and the users here
  write Finnish.
- **Targets:** Postgres `e73841f` and CPython `aa54070`, the pinned revisions in
  the suites' `tag.json`, shallow-fetched into `targets/` (gitignored).
- **jegrep:** release v0.1.2 Windows binary, SHA-256 checked against the release's
  `SHA256SUMS`. Default `cascade` strategy, TypeSafe endpoint, `jev-latest`, query
  only: the annotated keywords are never passed.
- **Index:** every `.c .h .py .y .l .go` and README file in 80-line windows,
  OpenAI `text-embedding-3-large` at 1024 dims, path on the first line.
  54,241 windows, 37M tokens, $4.81 one-off.
- **Rankers over the index:** BM25, cosine, and cosine top 30 reranked by Jev
  (one Noul per window, ten windows per request).

## Results

Twenty queries per language. File recall is macro-averaged; a ranker is scored on
as many files as jegrep returned for the same query, and on as many lines for
region recall, so a whole-file range cannot win by size.

| | English file recall | English regions | Finnish file recall | Finnish regions | $/query | s/query |
|---|---|---|---|---|---|---|
| jegrep, no index | **100%** | **100%** | 42% | 36% | $0.0049 | 4.0 |
| BM25 | 54% | 55% | 0% | 4% | ≈0 | <0.1 |
| embedding index | 88% | 89% | 52% | 41% | ≈0 | <0.1 |
| index + Jev rerank | 96% | 89% | **74%** | **58%** | $0.0012 | 0.4 |

Ranked, without jegrep's budget, the reranked index reaches 94% of files in its
top 5 in English and 90% in Finnish; plain cosine 80% and 66%.

## What it means

- **In English, on these suites, jegrep is the best result here**, with no index
  at all. But Postgres and CPython were jegrep's development suites, so its numbers
  are in-sample; its README reports the held-out Kubernetes suite at 17/17 files
  and 33/34 regions. It also returns a lot: 4,567 lines per query on average, and
  37 of 181 ranges were whole files.
- **In Finnish it breaks.** Five of twenty queries returned nothing. The cascade
  builds its candidates from a keyword scan of the query, and Finnish words do not
  occur in English code, so Jev never sees the right files. The judgment is not the
  weak part; the candidate step is. Translate the query, or pass English
  keywords with `-k`, before relying on it across languages.
- **An index plus a Jev rerank is the robust option** when queries and code do not
  share a language, and it is 10x faster per query. The rerank added 14 points in
  English and 24 in Finnish at 5 files. The price is the index: $4.81 to build
  here and re-embedding whatever changes. At these rates jegrep's per-query cost
  pays for the index after roughly 1,300 queries.
- **Lexical search alone is not a baseline worth keeping** for natural-language
  questions about code: 20% at rank 1 in English, nothing in Finnish.

## Limits

Twenty queries per language; a re-run moved rank-1 recall by one query. Labels
are positive-only, so extra files are unjudged, not wrong. The Finnish
translations are one person's. Region scores compare jegrep's top three ranges
per file with the ranker's windows at the same line count, not jegrep's own
metric exactly. Voyage `voyage-code-3` was the intended code embedding; this
account's rate limit made the index a four-hour job, so OpenAI's model was used.

## Run

```sh
uv sync
uv run python src/index.py postgres cpython   # paid: ~$4.8 of embeddings
uv run python src/run_jegrep.py               # paid: ~$0.16 of Jev
uv run python src/search.py                   # paid: ~$0.05 of Jev
uv run python src/score.py
```

Fetch `targets/` at the revisions above and put the jegrep release in `bin/`
first. Keys come from `~/.agents/env/_common.env` into memory only.
