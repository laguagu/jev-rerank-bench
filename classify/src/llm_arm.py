"""Chat-model arms, asked the same question over the same labels as Jev.

    uv run python src/llm_arm.py <arm> [dataset ...]

An arm is `<model>-<variant>`: `zs` (label names), `desc` (the induced label
definitions jev-desc uses) or `fewshot` (the ten nearest labelled messages).

| model | deployment | answers | probability |
| --- | --- | --- | --- |
| luna | Azure gpt-5.6-luna, reasoning none | label | none |
| sol | OpenAI gpt-5.6-sol, reasoning low | label | none |
| sol6 | OpenAI gpt-6-sol, reasoning low | label | none |
| mini | Azure gpt-4.1-mini | option number | top-20 logprobs over the numbers |
| luna6 | Azure gpt-6-luna, reasoning none | option number | the chosen token's logprob (the only one returned) |
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass

from common import (DATASETS, INSTRUCTIONS, Dataset, definitions, fewshot_note, load, load_env, neighbours,
                    post_json, top10)
from runner import run_arm

# USD per million tokens (input, output), OpenAI standard tier, short context,
# developers.openai.com/api/docs/pricing on 2026-09-24. Azure bills the same list prices.
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.6-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.00, 10.00),
}


@dataclass(frozen=True)
class Model:
    name: str
    direct: bool          # OpenAI API rather than the Azure deployment of the same name
    numbered: bool        # answers an option number instead of a label
    body: dict            # request parameters beyond the messages


MODELS = {
    "luna": Model("gpt-5.6-luna", False, False, {"max_completion_tokens": 64, "reasoning_effort": "none"}),
    "sol": Model("gpt-5.6-sol", True, False, {"max_completion_tokens": 2048, "reasoning_effort": "low"}),
    "sol6": Model("gpt-6-sol", True, False, {"max_completion_tokens": 2048, "reasoning_effort": "low"}),
    "mini": Model("gpt-4.1-mini", False, True, {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}),
    "luna6": Model("gpt-6-luna", False, True, {"max_completion_tokens": 8, "reasoning_effort": "none", "logprobs": True}),
}
AZURE_VERSION = {"gpt-5.6-luna": "2024-12-01-preview"}  # every other deployment answers on 2024-10-21


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def chat(model: Model, messages: list[dict]) -> dict:
    if model.direct:
        return post_json("https://api.openai.com/v1/chat/completions",
                         {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                         {"model": model.name, "messages": messages, **model.body})
    return post_json(f"https://{os.environ['AZURE_RESOURCE_NAME']}.openai.azure.com/openai/deployments/"
                     f"{model.name}/chat/completions?api-version={AZURE_VERSION.get(model.name, '2024-10-21')}",
                     {"api-key": os.environ["AZURE_API_KEY"]}, {"messages": messages, **model.body})


def render(d: Dataset, numbered: bool, examples: list[dict] | None, with_descriptions: bool) -> str:
    lines = [f"Context: {d.domain}", INSTRUCTIONS, "", "Intents:"]
    for i, (label, text) in enumerate(definitions(d, d.labels, with_descriptions).items()):
        desc = f": {text}" if text else ""
        lines.append(f"{i}. {label}{desc}" if numbered else f"- {label}{desc}")
    if examples:
        lines += ["", fewshot_note("The list below"), ""]
        lines += [f'- "{e["text"]}" -> {e["intent"]}' for e in examples]
    lines += ["", "Reply with the number of the intent only." if numbered
              else "Reply with the intent label only, exactly as written above."]
    return "\n".join(lines)


def parse_label(text: str, labels: list[str]) -> str | None:
    # Case and trailing punctuation are ignored on both sides: two BANKING77
    # labels are written `Refund_not_showing_up` and `reverted_card_payment?`.
    def norm(x: str) -> str:
        return x.strip().strip("`'\".?!").strip().lower()

    by_norm = {norm(l): l for l in labels}
    t = re.sub(r"^(intent|label)\s*:\s*", "", norm(text))
    for cand in (t, t.replace(" ", "_"), t.split()[0] if t else t):
        if norm(cand) in by_norm:
            return by_norm[norm(cand)]
    return None


def option(token: str, labels: list[str]) -> str | None:
    token = token.strip()
    return labels[int(token)] if token.isascii() and token.isdigit() and int(token) < len(labels) else None


def make(arm: str, ds: str):
    kind, variant = arm.split("-", 1)
    model = MODELS[kind]
    d = load(ds)
    examples = neighbours(d) if variant == "fewshot" else {}

    def fn(ex):
        system = render(d, model.numbered, examples.get(ex.id), with_descriptions=variant == "desc")
        j = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": ex.text}])
        choice = j["choices"][0]
        text = choice["message"]["content"] or ""
        i, o = j["usage"]["prompt_tokens"], j["usage"]["completion_tokens"]
        row = {"model": model.name, "in_tok": i, "out_tok": o, "usd": cost(model.name, i, o)}
        if not model.numbered:
            pred = parse_label(text, d.labels)
            return {"pred": pred, "raw": None if pred else text[:60], **row}
        first = (choice.get("logprobs") or {}).get("content") or [{}]
        if kind == "luna6":
            pred = option(text, d.labels)
            conf = math.exp(first[0]["logprob"]) if pred and "logprob" in first[0] else None
            return {"pred": pred, "raw": None if pred else text.strip()[:60], "conf": conf, **row}
        mass: dict[str, float] = {}
        for t in first[0].get("top_logprobs", []):
            if label := option(t["token"], d.labels):
                mass[label] = mass.get(label, 0) + math.exp(t["logprob"])
        total = sum(mass.values())
        ranked = sorted(((k, v / total) for k, v in mass.items()), key=lambda kv: -kv[1]) if total else []
        return {"pred": ranked[0][0] if ranked else None, "conf": ranked[0][1] if ranked else 0.0,
                "top": top10(ranked), **row}

    return d, fn


if __name__ == "__main__":
    load_env()
    arm = sys.argv[1]
    for ds in sys.argv[2:] or DATASETS:
        d, fn = make(arm, ds)
        run_arm(d, arm, fn, workers=6)
