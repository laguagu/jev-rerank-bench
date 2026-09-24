# Classification: is Jev the best intent classifier?

Seventeen classifiers
answer the same 600 messages on three public datasets, with the same question and
the same label list. Full tables: [results/report.md](results/report.md).

| Dataset | Labels | Language | Test messages |
| --- | --- | --- | --- |
| [BANKING77](https://huggingface.co/datasets/mteb/banking77) | 77 fine-grained banking intents | English | 600 of 3,080 |
| [CLINC150 plus](https://huggingface.co/datasets/clinc/clinc_oos) | 150 intents + out-of-scope | English | 600 of 5,500 (99 out-of-scope) |
| [MASSIVE fi](https://huggingface.co/datasets/mteb/amazon_massive_intent) | 60 smart-home intents | Finnish | 600 of 2,974 |

## Accuracy

| | BANKING77 | CLINC150 | MASSIVE fi | $/1000 | p50 latency |
| --- | --- | --- | --- | --- | --- |
| **No training data** | | | | | |
| Jev, label names | 77.8% | 90.5% | 80.3% | $0.03–0.06 | 0.24 s |
| gpt-5.6-luna, label names | 84.0% | 91.7% | 79.5% | $0.08–0.15 | 2.0 s |
| gpt-5.6-sol (low reasoning), label names | 86.3% | 94.5% | 82.5% | $0.96–1.61 | 1.8 s |
| gpt-4.1-mini, label names | 73.5% | 80.8% | 74.7% | $0.17–0.35 | 2.0 s |
| gpt-6-luna, label names | 76.8% | 81.0% | 71.7% | $0.04–0.09 | 1.6 s |
| gpt-6-sol (low reasoning), label names | 82.7% | 94.2% | 83.5% | $0.91–1.57 | 1.7 s |
| **Definitions induced from 8 examples per label** | | | | | |
| Jev | 87.0% | 94.2% | 84.2% | $0.10–0.19 | 0.25 s |
| gpt-5.6-luna | 89.7% | 93.2% | 86.2% | $0.33–0.61 | 2.3 s |
| **Ten nearest labelled messages in state** | | | | | |
| Jev, all labels | 93.0% | 94.7% | 89.0% | $0.05–0.08 | 0.24 s |
| Jev, embedding top-10 labels only | 93.7% | 94.8% | 89.0% | $0.03 | 0.24 s |
| gpt-5.6-luna | 94.2% | 94.5% | 90.2% | $0.12–0.19 | 2.1 s |
| gpt-6-luna | 93.5% | 90.2% | 85.3% | $0.06–0.11 | 1.5 s |
| **Trained on the full training split** | | | | | |
| logistic regression on embeddings | 94.2% | 93.7% | 89.2% | $0.002 | ms |
| TF-IDF + logistic regression | 90.5% | 83.3% | 83.3% | $0 | ms |

95% intervals are about ±3 points at n=600. Paired McNemar tests are in the report.
Chat-model latency includes a new HTTPS connection per call (the harness now pools
connections; the committed rows predate that), an estimated 0.05–0.15 s.

## What it shows

- **Not the most accurate.** With label names only, gpt-5.6-sol beat Jev by 8.5
  points on BANKING77 and 4 on CLINC (both p < 0.001) and tied on Finnish; gpt-6-sol
  beat it on all three (p < 0.02). gpt-5.6-luna also beat it on BANKING77.
- **The newest small model is no substitute.** gpt-6-luna, at $0.10 per million input
  tokens, was 9 points behind Jev on CLINC and Finnish (p < 0.001) and level on BANKING77.
  With examples in state Jev was also ahead on CLINC and Finnish (p < 0.002).
- **Wording closes most of the gap.** Most zero-shot errors were label names that
  do not separate neighbours (`order_physical_card` → `get_physical_card`, 11 times).
  One-sentence definitions cut BANKING77 errors from 133 to 78 and put Jev level with
  gpt-5.6-sol at about a tenth of the price.
- **With labelled data, nothing beat a trained classifier.** Every arm given the ten
  nearest labelled messages tied logistic regression on embeddings, which costs
  almost nothing and answers in milliseconds.
- **Where Jev is clearly ahead:** 8x lower latency than the chat models, about 30x lower
  cost than gpt-5.6-sol on the same label list, and a probability that gates. With examples in state it
  answered 97% / 100% / 88% of messages automatically at 95% accuracy, on par with the
  trained classifier. gpt-6-luna with the same examples managed 93% / 90% / 81% from its
  single returned logprob; gpt-4.1-mini's logprobs 22% / 61% / 12% zero-shot; GPT-5.x
  returns no probability at all.
- **Out-of-scope without examples:** Jev caught 88% of CLINC's out-of-scope requests at
  90% precision zero-shot, against 72% for the trained classifier. Only the sol models
  did better (93–94%). Adding definitions or examples lowered it to 74–79%: neighbours pull
  towards an in-scope label.
- **Finnish is not a weakness:** zero-shot Jev tied both chat models on MASSIVE fi.

## Limits

One run per arm, 600 messages per dataset, public intent sets whose label names are
already informative. The sol arms used low reasoning effort; more reasoning could raise
them. Claude arms were planned, but the API key had no credit on the day. The
definitions were written by gpt-5.6-sol from training examples, so those arms are not
zero-shot.

## Run

```sh
uv sync
OPENBLAS_NUM_THREADS=1 uv run python src/local_arms.py   # embeddings ~$0.06
uv run python src/describe.py                            # ~$0.14
uv run python src/jev_arm.py jev-zs                      # also jev-desc, jev-fewshot, jev-hybrid
uv run python src/llm_arm.py luna-zs                     # also luna-desc, luna-fewshot, mini-*, sol-zs, luna6-*, sol6-zs
uv run python src/report.py
```

Download the parquet and JSON files listed in `src/common.py` into `data/` first,
and run `local_arms.py` before the few-shot and hybrid arms: it writes the nearest
neighbours they read. `results/*.jsonl` holds every answer, so the report re-scores
without another call. Everything here cost about $7.90, of which Jev was $0.50.
