# classify — conventions

Is Jev the best intent classifier? Read `README.md` for the question and the
numbers, `results/report.md` for the full tables. An experiment, not a product:
anything that weakens the comparison matters more than tidy code.

## Things that will bite

- **Every arm answers the identical sample.** `common.load()` shuffles the test
  split with a fixed seed and keeps 600. Changing `SAMPLE_SIZE` or `SEED` changes
  which messages every arm saw; delete `results/*.jsonl` when you do.
- **One question, one label list.** `INSTRUCTIONS`, `fewshot_note`, `definitions`
  and `OOS_DESCRIPTION` live in `common.py` and are used by the Jev and chat arms
  alike; `humanize` is for the embedding zero-shot only. Restate the wording in one
  arm and you are comparing prompts. A refactor must leave `in_tok` unchanged for a
  stored message: that is the check that the prompt is byte-identical.
- **Two BANKING77 labels are odd:** `Refund_not_showing_up` (capital R) and
  `reverted_card_payment?`. `llm_arm.parse_label` ignores case and trailing
  punctuation; before it did, correct chat answers were scored as failures.
  `report.py` re-parses saved `raw` answers, so old rows stay usable.
- **Result files are append-only and resumable.** A row with `error` is retried
  on the next run; `read_rows` keeps the first successful row per id.
- **Gate Jev on the chosen label's probability (`p_pred`)**, not `confidence`,
  when comparing with other arms' max probability. They are different signals.
- **Memory.** The machine this ran on had under 1 GB of free commit memory with
  many agent sessions open. TF-IDF uses `saga` and capped features, and BLAS runs
  single-threaded (`OPENBLAS_NUM_THREADS=1`); lbfgs on char n-grams did not fit.
- **Paid calls.** Jev ($0.042/M input) and the embeddings are cents. The sol arms
  are the expensive ones (~$1 to $1.6 per 600 messages).

## Secrets

Keys are read into memory from `~/.agents/env/_common.env` by `common.load_env`
(only the names in `_WANTED`), and only by the scripts that call a provider. Nothing is written to the repo. Provider error
bodies are never logged, because they can echo input.

## Data

Public Hugging Face datasets, downloaded, not vendored: `mteb/banking77`
(CC-BY-4.0), `clinc/clinc_oos` plus (CC-BY-3.0), `mteb/amazon_massive_intent`
fi (Apache-2.0). URLs are in `README.md`.
