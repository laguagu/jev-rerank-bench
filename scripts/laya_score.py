"""Score every (query, candidate) pair with Laya and write the scores to JSON.

Laya (Apache-2.0, `convaiinnovations/laya`) answers the same three primitives as
Jev — choice, score, noul — from local weights in a single forward pass. That
makes it the open-weight control for this benchmark: if it matches a hosted
typed-judgment model on reranking, the capability is not something you have to
buy.

It runs in Python, so it cannot import `src/rerank/questions.ts`. It reads the
dumped spec instead (`bun run dump-question`), which keeps the wording in one
place — the whole benchmark rests on every model being asked the same question.

    bun run dump-question --dataset=mupler
    .venv-laya/Scripts/python.exe scripts/laya_score.py --dataset mupler

Output: `<dataset dir>/laya-scores.json`, keyed by query id, then chunk id, ready
for `--reranker=laya` in the benchmark. Nothing here calls a paid API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import laya

DATASETS = {"fi-tes": Path("data/fi-tes"), "mupler": Path("data/mupler")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fi-tes", choices=sorted(DATASETS))
    ap.add_argument("--stage", default="hybrid", help="which first-stage shortlist to rerank")
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument(
        "--subfolder",
        default="multilingual",
        help="Laya checkpoint. Both corpora are Finnish, so the multilingual one is the right default.",
    )
    ap.add_argument("--kind", default="noul", choices=["noul", "score"])
    ap.add_argument("--limit", type=int, default=0, help="smoke-test against the first N queries")
    args = ap.parse_args()

    root = DATASETS[args.dataset]
    corpus = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    shortlists = json.loads((root / f"shortlists-k{args.candidates}.json").read_text(encoding="utf-8"))
    spec = json.loads((root / "question-spec.json").read_text(encoding="utf-8"))

    queries = corpus["queries"]
    if args.limit:
        queries = queries[: args.limit]
    lists = shortlists[args.stage][: len(queries)]

    if args.kind == "noul":
        question = {
            "relevant": {
                "type": "noul",
                "instructions": "\n".join(str(v) for v in spec["instructions"].values()),
                "criteria": {"true": spec["criteria"]["true"], "false": spec["criteria"]["false"]},
            }
        }
    else:
        question = {
            "relevant": {
                "type": "score",
                "instructions": "\n".join(str(v) for v in spec["instructions"].values()),
                "criteria": list(spec["rubric"]),
            }
        }

    print(f"dataset {args.dataset}, stage {args.stage}, {len(queries)} queries x {args.candidates} candidates")
    print(f"loading laya ({args.subfolder}) …")
    t0 = time.time()
    agent = laya.load("convaiinnovations/laya", subfolder=args.subfolder)
    print(f"  loaded in {time.time() - t0:.1f}s")

    scores: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    failures = 0
    start = time.time()

    for qi, q in enumerate(queries):
        per_query: dict[str, float] = {}
        t_q = time.time()
        for cand in lists[qi]["candidates"]:
            # The same fields the Jev state carries for this corpus: MuPLeR
            # passages have no meaningful document id, so only the text is sent.
            state = {"question": q["question"], "passage": cand["body"]}
            if args.dataset == "fi-tes":
                state["document"] = cand["docTitle"]
                state["section"] = cand["headingPath"] or cand["heading"]
            try:
                out = agent.predict(state, question)
                ans = out["answers"]["relevant"]
                per_query[str(cand["chunkId"])] = float(ans["noul"] if args.kind == "noul" else ans["score"])
            except Exception as exc:  # noqa: BLE001 - one bad pair must not lose the run
                failures += 1
                if failures <= 3:
                    print(f"  ! {type(exc).__name__}: {str(exc)[:120]}")
        latencies.append((time.time() - t_q) * 1000)
        scores[q["id"]] = per_query
        if (qi + 1) % 10 == 0 or qi + 1 == len(queries):
            done = time.time() - start
            rate = (qi + 1) / done
            print(f"  {qi + 1}/{len(queries)} queries  {done:.0f}s  eta {(len(queries) - qi - 1) / rate:.0f}s")

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    out_path = root / f"laya-scores-{args.kind}.json"
    out_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "stage": args.stage,
                "kind": args.kind,
                "subfolder": args.subfolder,
                "candidates": args.candidates,
                "perQueryMsP50": p50,
                "perQueryMsP95": p95,
                "failures": failures,
                "scores": scores,
            }
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    print(f"  per-query wall time p50 {p50:.0f} ms, p95 {p95:.0f} ms, {failures} failures")
    print(f"  total {time.time() - start:.0f}s for {len(queries) * args.candidates} decisions")


if __name__ == "__main__":
    main()
