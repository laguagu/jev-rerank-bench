"""Scores every arm on the identical sample and writes results/report.md.

    uv run python src/report.py

- accuracy with a Wilson 95% interval; a failed or unparseable answer counts as wrong
- macro-F1 over the labels that occur in the sample
- CLINC only: accuracy on in-scope messages, recall and precision of out-of-scope
- gating, for arms that return a probability: the largest share of messages that
  can be answered automatically, most confident first, while accuracy on that share
  stays at 95% or 98% (a ranking measure, identical for every arm). The same
  threshold chosen on one half of the sample and replayed on the other is kept in
  summary.json as a check on how well a fixed threshold transfers
- ECE over ten bins; median latency per call; USD per 1000 messages
- McNemar exact test for the paired comparisons named in PAIRS
"""

from __future__ import annotations

import json
import statistics

from scipy.stats import binomtest
from sklearn.metrics import f1_score

from common import DATASETS, RESULTS, load, read_rows, result_path
from embed import USD_PER_M as EMBED_USD_PER_M
from llm_arm import parse_label

ARMS = ["tfidf-lr", "emb-lr", "emb-knn10", "emb-zeroshot",
        "jev-zs", "jev-desc", "jev-fewshot", "jev-hybrid",
        "luna-zs", "luna-desc", "luna-fewshot", "mini-zs", "mini-fewshot", "sol-zs",
        "luna6-zs", "luna6-fewshot", "sol6-zs"]
TRAINING = {"tfidf-lr": "yes", "emb-lr": "yes", "emb-knn10": "yes",
            "jev-fewshot": "10 nearest", "jev-hybrid": "10 nearest", "luna-fewshot": "10 nearest",
            "mini-fewshot": "10 nearest", "luna6-fewshot": "10 nearest",
            "jev-desc": "definitions from 8/label", "luna-desc": "definitions from 8/label"}
NO_LATENCY = {"tfidf-lr", "emb-lr", "emb-knn10", "emb-zeroshot"}
PAIRS = [("jev-zs", "sol6-zs"), ("jev-zs", "luna6-zs"), ("jev-zs", "sol-zs"), ("jev-zs", "luna-zs"),
         ("jev-desc", "jev-zs"), ("jev-desc", "luna-desc"), ("jev-fewshot", "luna-fewshot"),
         ("jev-hybrid", "luna6-fewshot"), ("jev-hybrid", "emb-lr"), ("jev-fewshot", "emb-lr")]
EMBED_USD_PER_1000_QUERIES = EMBED_USD_PER_M * 12 / 1000  # ~12 tokens per message


def correct(r: dict) -> bool:
    return r.get("pred") == r["gold"]


def gate(rows: list[dict], order: dict[str, int]) -> dict | None:
    # Jev reports both the chosen label's probability and a distribution-level
    # confidence; the chosen probability is the one comparable to the other arms.
    conf = {r["id"]: r.get("p_pred", r.get("conf")) for r in rows}
    if sum(c is not None for c in conf.values()) < len(rows) * 0.95:
        return None
    conf = {k: c or 0.0 for k, c in conf.items()}
    by = sorted(rows, key=lambda r: -conf[r["id"]])

    def cov_at(target):
        best, right = 0, 0
        for i, r in enumerate(by, 1):
            right += correct(r)
            if right / i >= target:
                best = i
        return best / len(by)

    # Threshold for >=95% accuracy chosen on the even half, reported on the odd half.
    dev = [r for r in rows if order[r["id"]] % 2 == 0]
    test = [r for r in rows if order[r["id"]] % 2 == 1]
    thr = None
    for t in sorted({round(conf[r["id"]], 4) for r in dev}):
        acc = [correct(r) for r in dev if conf[r["id"]] >= t]
        if acc and sum(acc) / len(acc) >= 0.95:
            thr = t
            break
    kept = [r for r in test if thr is not None and conf[r["id"]] >= thr]

    bins: list[list[dict]] = [[] for _ in range(10)]
    for r in rows:
        bins[int(min(0.9999, max(0.0, conf[r["id"]])) * 10)].append(r)
    ece = sum(len(b) / len(rows) * abs(sum(map(correct, b)) / len(b) - statistics.mean(conf[r["id"]] for r in b))
              for b in bins if b)
    top80 = by[: round(len(by) * 0.8)]
    return {"acc80": sum(map(correct, top80)) / len(top80), "cov95": cov_at(0.95), "cov98": cov_at(0.98),
            "heldout_cov95": len(kept) / len(test), "heldout_acc95": sum(map(correct, kept)) / len(kept) if kept else 0.0,
            "thr95": thr, "ece": ece}


def mcnemar(a: list[dict], b: list[dict]) -> tuple[int, int, float]:
    ma = {r["id"]: correct(r) for r in a}
    mb = {r["id"]: correct(r) for r in b}
    ids = ma.keys() & mb.keys()
    only_a = sum(ma[i] and not mb[i] for i in ids)
    only_b = sum(mb[i] and not ma[i] for i in ids)
    p = binomtest(min(only_a, only_b), only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(p)


def main() -> None:
    summary: dict = {}
    out = ["# Results", ""]
    for ds in DATASETS:
        d = load(ds)
        order = {e.id: i for i, e in enumerate(d.test)}
        columns = ["arm", "uses train data", "acc (95% CI)", "macro-F1"]
        if d.oos_label:
            columns += ["in-scope acc", "OOS recall / precision"]
        columns += ["auto-answered at 95% / 98% acc", "ECE", "p50 ms", "$/1000"]
        out += [f"## {ds} — {len(d.labels)} labels, n={len(d.test)}", "",
                "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        summary[ds] = {}
        rows_by_arm = {}
        for arm in ARMS:
            rows = [r for r in read_rows(result_path(ds, arm)) if r["id"] in order]
            if len(rows) < len(d.test):
                continue
            for r in rows:  # answers saved before the label parser was fixed are parsed again
                if r.get("pred") is None and r.get("raw"):
                    r["pred"] = parse_label(r["raw"], d.labels)
            rows_by_arm[arm] = rows
            k = sum(map(correct, rows))
            ci = binomtest(k, len(rows)).proportion_ci(method="wilson")
            f1 = f1_score([r["gold"] for r in rows], [r.get("pred") or "" for r in rows],
                          labels=sorted({r["gold"] for r in rows}), average="macro")
            g = gate(rows, order)
            ms = 0 if arm in NO_LATENCY else statistics.median(r.get("ms", 0) for r in rows)
            usd = sum(r.get("usd", 0) for r in rows) / len(rows) * 1000
            if arm.startswith("emb") or arm == "jev-hybrid":
                usd += EMBED_USD_PER_1000_QUERIES
            errors = sum(1 for r in rows if r.get("error") or r.get("pred") is None)
            s = {"acc": k / len(rows), "ci": [ci.low, ci.high], "f1": f1, "gate": g, "ms": ms,
                 "usd_per_1000": usd, "errors": errors,
                 "models": sorted({r["model"] for r in rows if r.get("model")})}
            cells = [f"`{arm}`", TRAINING.get(arm, "no"),
                     f"**{k / len(rows):.1%}** ({ci.low:.1%}–{ci.high:.1%})" + (f" · {errors} failed" if errors else ""),
                     f"{f1:.3f}"]
            if d.oos_label:
                ins = [r for r in rows if r["gold"] != d.oos_label]
                oos = [r for r in rows if r["gold"] == d.oos_label]
                pred_oos = [r for r in rows if r.get("pred") == d.oos_label]
                ins_acc = sum(map(correct, ins)) / len(ins)
                rec = sum(map(correct, oos)) / len(oos)
                prec = sum(map(correct, pred_oos)) / len(pred_oos) if pred_oos else 0.0
                s.update({"in_scope_acc": ins_acc, "oos_recall": rec, "oos_precision": prec, "n_oos": len(oos)})
                cells += [f"{ins_acc:.1%}", f"{rec:.0%} / {prec:.0%}"]
            cells += [f"{g['cov95']:.0%} / {g['cov98']:.0%}", f"{g['ece']:.3f}"] if g else ["no probability", "—"]
            cells += [f"{ms:.0f}" if ms else "—", f"${usd:.3f}"]
            out.append("| " + " | ".join(cells) + " |")
            summary[ds][arm] = s
        out += ["", "Paired (McNemar exact): right only in A / right only in B, p", ""]
        for a, b in PAIRS:
            if a in rows_by_arm and b in rows_by_arm:
                x, y, p = mcnemar(rows_by_arm[a], rows_by_arm[b])
                out.append(f"- `{a}` vs `{b}`: {x} / {y}, p = {p:.3g}")
                summary[ds].setdefault("pairs", {})[f"{a} vs {b}"] = [x, y, p]
        out.append("")
    (RESULTS / "report.md").write_text("\n".join(out), encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
