"""Jev arms: one Choice per message over the dataset's labels.

    uv run python src/jev_arm.py <arm> [dataset ...]

Arms
- jev-zs        every label, label names only (the pure zero-shot case)
- jev-desc      every label with a one-sentence definition induced from eight
                training examples per label (src/describe.py)
- jev-fewshot   every label, plus the ten nearest labelled training messages in state
- jev-hybrid    only the embedding classifier's top ten labels (plus out-of-scope),
                with the same ten examples: retrieval narrows, Jev decides
"""

from __future__ import annotations

import sys

from common import (DATASETS, INSTRUCTIONS, definitions, fewshot_note, load, load_env, neighbours,
                    read_rows, result_path, top10)
from runner import run_arm

USD_PER_M_INPUT = 0.042  # docs.typesafe.ai/models, 2026-09-24; output tokens are free
ARMS = ("jev-zs", "jev-desc", "jev-fewshot", "jev-hybrid")


def make(arm: str, ds: str, client):
    from typesafe_sdk import Choice

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    d = load(ds)
    examples = neighbours(d) if arm in ("jev-fewshot", "jev-hybrid") else {}
    shortlist = ({r["id"]: [t[0] for t in r["top"]] for r in read_rows(result_path(ds, "emb-lr"))}
                 if arm == "jev-hybrid" else {})
    instructions = f"{INSTRUCTIONS} {fewshot_note('`labelled_examples`')}" if examples else INSTRUCTIONS

    def fn(ex):
        state = {"context": d.domain, "message": ex.text}
        labels = d.labels
        if examples:
            state["labelled_examples"] = examples[ex.id]
        if shortlist:
            labels = shortlist[ex.id] + ([d.oos_label] if d.oos_label and d.oos_label not in shortlist[ex.id] else [])
        criteria = definitions(d, labels, with_descriptions=arm == "jev-desc")
        r = client.system_one(state=state, questions={"intent": Choice(instructions=instructions, criteria=criteria)})
        a = r.answers["intent"]
        return {"pred": a.choice, "conf": a.confidence, "p_pred": a.probabilities[a.choice],
                "top": top10(a.probabilities.items()), "model": r.model,
                "in_tok": r.usage.input_tokens, "out_tok": r.usage.output_tokens,
                "usd": r.usage.input_tokens / 1e6 * USD_PER_M_INPUT}

    return d, fn


if __name__ == "__main__":
    load_env()
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    client = TypeSafeClient(model="jev-latest", retry=RetryPolicy(max_retries=4), timeout=60)
    arm = sys.argv[1]
    for ds in sys.argv[2:] or DATASETS:
        d, fn = make(arm, ds, client)
        run_arm(d, arm, fn, workers=8)
