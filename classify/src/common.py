"""Shared plumbing: secrets, datasets, the test sample, the wording every model
sees, one HTTP helper, and result files.

Every arm reads the same sample, the same label list and the same question from
here, so a difference in the metrics can only come from the classifier.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SAMPLE_SIZE = 600
SEED = 20260924

# Keys are read from the synced agent config into memory only. Nothing is
# written to the repository, and only the names below are ever loaded.
_ENV_FILE = Path.home() / ".agents" / "env" / "_common.env"
_WANTED = {"TYPESAFE_API_KEY", "OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_RESOURCE_NAME"}


def load_env() -> None:
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) in _WANTED and not os.environ.get(m.group(1)):
            os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


# One pooled client for every provider call, so chat arms do not pay a new TLS
# handshake per request while Jev's SDK reuses its connections.
_http = httpx.Client(timeout=120, limits=httpx.Limits(max_connections=16))


def post_json(url: str, headers: dict, body: dict) -> dict:
    """POST with backoff on 429 and 5xx. Other failures raise with the status
    only: provider error bodies can echo the input."""
    for attempt in range(6):
        r = _http.post(url, headers=headers, json=body)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30, float(r.headers.get("retry-after", 2 ** attempt))))
            continue
        raise RuntimeError(f"HTTP {r.status_code}")
    raise RuntimeError("retries exhausted")


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str


@dataclass(frozen=True)
class Dataset:
    name: str
    language: str
    labels: list[str]          # every label an arm may answer, including the out-of-scope one
    oos_label: str | None      # the label that means "none of these", if the task has one
    train: list[Example]
    test: list[Example]        # the fixed evaluation sample, identical for every arm
    domain: str                # one line of context every arm receives


def _parquet(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def _jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sample(rows: list[Example], n: int) -> list[Example]:
    rng = random.Random(SEED)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def load(name: str, n: int = SAMPLE_SIZE) -> Dataset:
    if name == "banking77":
        tr = [Example(f"tr{i}", r["text"], r["label_text"]) for i, r in enumerate(_parquet(DATA / "banking77-train.parquet"))]
        te = [Example(f"te{i}", r["text"], r["label_text"]) for i, r in enumerate(_parquet(DATA / "banking77-test.parquet"))]
        labels = sorted({e.label for e in tr})
        return Dataset(name, "en", labels, None, tr, _sample(te, n),
                       "Customer messages sent to an online bank's support chat.")
    if name == "clinc":
        schema = pq.read_schema(DATA / "clinc-test.parquet")
        names = json.loads(schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]

        def rows(file: str, prefix: str) -> list[Example]:
            return [Example(f"{prefix}{i}", r["text"], names[r["intent"]]) for i, r in enumerate(_parquet(DATA / file))]

        tr = rows("clinc-train.parquet", "tr")
        te = rows("clinc-test.parquet", "te")
        labels = sorted(set(names))
        return Dataset(name, "en", labels, "oos", tr, _sample(te, n),
                       "Requests spoken to a general-purpose voice assistant (banking, travel, home, work, small talk and more).")
    if name == "massive-fi":
        tr = [Example(f"tr{r['id']}", r["text"], r["label"]) for r in _jsonl_gz(DATA / "massive-fi-train.json.gz")]
        te = [Example(f"te{r['id']}", r["text"], r["label"]) for r in _jsonl_gz(DATA / "massive-fi-test.json.gz")]
        labels = sorted({e.label for e in tr} | {e.label for e in te})
        return Dataset(name, "fi", labels, None, tr, _sample(te, n),
                       "Finnish commands and questions spoken to a smart-home voice assistant.")
    raise ValueError(f"unknown dataset {name}")


DATASETS = ["banking77", "clinc", "massive-fi"]

# ------------------------------------------------------------------ the wording
# Every Jev and chat arm renders its request from these, never from its own copy.

INSTRUCTIONS = "Which intent does the message express?"
OOS_DESCRIPTION = "None of the listed intents: the request is about something the assistant does not support."


def fewshot_note(subject: str) -> str:
    """Jev names the state field; a chat prompt says "The list below"."""
    return (f"{subject} holds similar messages that people have already labelled. "
            "They may or may not share this message's intent; judge the message itself.")


def definitions(d: Dataset, labels: list[str], with_descriptions: bool) -> dict[str, str | None]:
    """What each label says beside its name: an induced one-sentence definition
    (the *-desc arms), otherwise nothing except for the out-of-scope label."""
    if with_descriptions:
        described = json.loads((DATA / f"descriptions-{d.name}.json").read_text(encoding="utf-8"))
        return {l: described[l] for l in labels}
    return {l: (OOS_DESCRIPTION if l == d.oos_label else None) for l in labels}


def neighbours(d: Dataset) -> dict[str, list[dict]]:
    """The ten nearest labelled training messages per test id (local_arms.py)."""
    return json.loads((DATA / f"neighbours-{d.name}.json").read_text(encoding="utf-8"))


def humanize(label: str) -> str:
    """`card_arrival` -> `card arrival`, for the embedding zero-shot arm only;
    the Jev and chat arms see the label names as the dataset writes them."""
    return label.replace("_", " ")


def top10(pairs) -> list[list]:
    """(label, probability) pairs as stored in every result row."""
    return [[l, round(float(p), 4)] for l, p in sorted(pairs, key=lambda kv: -kv[1])[:10]]

# ---------------------------------------------------------------- result files


def result_path(dataset: str, arm: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"{dataset}__{arm}.jsonl"


def read_rows(path: Path) -> list[dict]:
    """One row per id: the last successful attempt, or the first failure when none succeeded."""
    if not path.exists():
        return []
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] not in rows or not row.get("error"):
                rows[row["id"]] = row
    return list(rows.values())


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_rows(path) if not r.get("error")}
