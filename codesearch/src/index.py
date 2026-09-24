"""Builds the embedding index that jegrep does without.

    uv run python src/index.py postgres cpython

Every tracked source file (.c .h .py .y .l .go and README*) is cut into
80-line windows; each window is embedded with OpenAI `text-embedding-3-large`
at 1024 dims, with its path on the first line. Vectors are stored as float16.
This index is the up-front cost an embedding search pays and jegrep does not,
so its token count and price are recorded next to it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "targets"
INDEX = ROOT / "index"
WINDOW = 80
MAX_CHARS = 6000
MODEL = "text-embedding-3-large"
USD_PER_M = 0.13
JEV_USD_PER_M = 0.042  # docs.typesafe.ai/models, 2026-09-24; output tokens are free
SOURCE = re.compile(r"\.(c|h|py|y|l|go)$|(^|/)README[^/]*$")


def load_env() -> None:
    for line in (Path.home() / ".agents" / "env" / "_common.env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(OPENAI_API_KEY|TYPESAFE_API_KEY)\s*=\s*(.*)$", line)
        if m and not os.environ.get(m.group(1)):
            os.environ[m.group(1)] = m.group(2).strip().strip('"')


def chunks(repo: str) -> list[dict]:
    root = TARGETS / repo
    files = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    out = []
    for f in files:
        if not f or not SOURCE.search(f):
            continue
        try:
            lines = (root / f).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for start in range(0, max(1, len(lines)), WINDOW):
            body = "\n".join(lines[start:start + WINDOW])
            if not body.strip():
                continue
            out.append({"file": f, "start": start + 1, "end": min(len(lines), start + WINDOW),
                        "text": f"path: {f}\n{body}"[:MAX_CHARS]})
    return out


def embed(texts: list[str]) -> tuple[np.ndarray, int]:
    for attempt in range(8):
        r = httpx.post("https://api.openai.com/v1/embeddings",
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       json={"input": texts, "model": MODEL, "dimensions": 1024}, timeout=120)
        if r.status_code == 200:
            j = r.json()
            v = np.array([d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"])], dtype=np.float32)
            return v, j["usage"]["total_tokens"]
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 2 ** attempt * 2))
            continue
        if r.status_code == 400:
            # A window of dense symbols can exceed the model's 8k-token input cap even
            # under MAX_CHARS. Split the batch, and halve a single text until it fits.
            if len(texts) > 1:
                parts = [embed([t]) for t in texts]
                return np.concatenate([p[0] for p in parts]), sum(p[1] for p in parts)
            if len(texts[0]) > 200:
                return embed([texts[0][: len(texts[0]) // 2]])
        raise RuntimeError(f"embeddings {r.status_code}")
    raise RuntimeError("embeddings: retries exhausted")


def build(repo: str) -> None:
    INDEX.mkdir(exist_ok=True)
    meta_path, vec_path = INDEX / f"{repo}.chunks.json", INDEX / f"{repo}.f16.npy"
    if vec_path.exists():
        print(f"{repo}: index exists", flush=True)
        return
    cs = chunks(repo)
    # 100 windows is ~70k tokens, well under the 300k-token request cap.
    batches = [cs[i:i + 100] for i in range(0, len(cs), 100)]
    started, done, tokens = time.perf_counter(), 0, 0
    vecs = [None] * len(batches)

    part_dir = INDEX / f"{repo}.parts"
    part_dir.mkdir(exist_ok=True)

    def work(i):
        nonlocal done, tokens
        part = part_dir / f"{i:05d}.npy"
        if part.exists():  # a finished batch is never paid for twice
            vecs[i] = np.load(part)
            done += 1
            return
        v, t = embed([c["text"] for c in batches[i]])
        vecs[i] = v.astype(np.float16)
        np.save(part, vecs[i])
        done += 1
        tokens += t
        if done % 100 == 0:
            print(f"{repo}: {done}/{len(batches)} batches, {tokens / 1e6:.1f}M tokens", flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(len(batches))))
    np.save(vec_path, np.concatenate(vecs))
    meta_path.write_text(json.dumps([{k: c[k] for k in ("file", "start", "end")} for c in cs]), encoding="utf-8")
    info = {"chunks": len(cs), "tokens": tokens, "usd": tokens / 1e6 * USD_PER_M,
            "seconds": round(time.perf_counter() - started), "model": MODEL}
    (INDEX / f"{repo}.build.json").write_text(json.dumps(info), encoding="utf-8")
    print(f"{repo}: {info}", flush=True)


if __name__ == "__main__":
    load_env()
    for repo in sys.argv[1:]:
        build(repo)
