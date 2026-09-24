"""Azure `text-embedding-3-large` at 1024 dimensions, cached per dataset.

The cache is content-addressed by the text list's hash, so a changed sample
re-embeds instead of silently reusing stale vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DATA, post_json

DIMS = 1024
DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
API_VERSION = "2024-02-01"
USD_PER_M = 0.13


def _batch(texts: list[str]) -> tuple[np.ndarray, int]:
    j = post_json(f"https://{os.environ['AZURE_RESOURCE_NAME']}.openai.azure.com/openai/deployments/"
                  f"{DEPLOYMENT}/embeddings?api-version={API_VERSION}",
                  {"api-key": os.environ["AZURE_API_KEY"]}, {"input": texts, "dimensions": DIMS})
    out = np.zeros((len(texts), DIMS), dtype=np.float32)
    for d in j["data"]:
        out[d["index"]] = d["embedding"]
    return out, j["usage"]["prompt_tokens"]


def embed(texts: list[str], tag: str) -> np.ndarray:
    key = hashlib.sha256(json.dumps(texts).encode()).hexdigest()[:16]
    path = DATA / f"emb-{tag}-{key}.npy"
    if path.exists():
        return np.load(path)
    chunks = [texts[i:i + 256] for i in range(0, len(texts), 256)]
    with ThreadPoolExecutor(2) as pool:
        parts = list(pool.map(_batch, chunks))
    vecs = np.concatenate([p[0] for p in parts])
    tokens = sum(p[1] for p in parts)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(path, vecs)
    (DATA / f"emb-{tag}-{key}.usage.json").write_text(json.dumps({"tokens": tokens, "usd": tokens / 1e6 * USD_PER_M}))
    return vecs
