"""Writes a one-sentence definition for every label from eight training examples.

    uv run python src/describe.py

Label names alone leave near-neighbours ambiguous (`card_arrival` against
`card_delivery_estimate`). This induces a definition per label with gpt-5.6-sol
so the `*-desc` arms can test whether wording, rather than the model, limits
zero-shot accuracy. It uses a little training data (8 per label), so those arms
sit between zero-shot and few-shot.
"""

from __future__ import annotations

import json
import random

from common import DATA, DATASETS, OOS_DESCRIPTION, load, load_env
from llm_arm import Model, chat, cost

# The writer of the definitions: gpt-5.6-sol at medium reasoning, answering JSON.
WRITER = Model("gpt-5.6-sol", True, False, {"response_format": {"type": "json_object"},
                                            "reasoning_effort": "medium", "max_completion_tokens": 32000})

PER_LABEL = 8


def describe(ds: str) -> None:
    path = DATA / f"descriptions-{ds}.json"
    if path.exists():
        return
    d = load(ds)
    rng = random.Random(7)
    by: dict[str, list[str]] = {}
    for e in d.train:
        by.setdefault(e.label, []).append(e.text)
    labels = [l for l in d.labels if l != d.oos_label]
    blocks = []
    for l in labels:
        ex = rng.sample(by[l], min(PER_LABEL, len(by[l])))
        blocks.append(f"## {l}\n" + "\n".join(f"- {t}" for t in ex))
    prompt = (f"These are intent labels for a classifier. Context: {d.domain}\n"
              "For every label, write one plain English sentence that defines which messages belong to it, "
              "written so it can be told apart from the most similar other labels in this list. "
              "Describe the user's need, not the wording of the examples. "
              "Return a JSON object mapping each label to its sentence, with every label present exactly once.\n\n"
              + "\n\n".join(blocks))
    j = chat(WRITER, [{"role": "user", "content": prompt}])
    out = json.loads(j["choices"][0]["message"]["content"])
    missing = [l for l in labels if l not in out]
    if missing:
        raise RuntimeError(f"{ds}: {len(missing)} labels without a description")
    out = {l: out[l] for l in labels}
    if d.oos_label:
        out[d.oos_label] = OOS_DESCRIPTION
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    usd = cost(WRITER.name, j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"])
    print(f"{ds}: {len(out)} descriptions, ${usd:.3f}", flush=True)


if __name__ == "__main__":
    load_env()
    for ds in DATASETS:
        describe(ds)
