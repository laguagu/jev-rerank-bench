"""Index-based retrieval on the same queries jegrep answers.

    uv run python src/search.py

Rankers, all over the same 80-line windows:
- bm25       lexical, identifiers split on case and underscores
- emb        text-embedding-3-large cosine, file score = best window
- emb+jev    the 30 best embedding windows judged by Jev, one Noul per window,
             ten windows per request; windows the model never scored keep
             their place behind the scored ones, so the shortlist is unchanged

Raw per-query rankings go to results/rankings-<lang>.json.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

from index import INDEX, JEV_USD_PER_M, MAX_CHARS, ROOT, TARGETS, embed, load_env
from run_jegrep import cases

SHORTLIST = 30
BATCH = 10

RELEVANCE = {
    "task": "A developer searched this code base with `query`. Decide whether the code window "
            "`windows[i]` is a place they would open to answer it.",
}
CRITERIA = {
    "true": "The window contains the code, or the comment block, that carries out or directly "
            "explains the mechanism the query asks about.",
    "false": "The window only mentions, calls, configures or tests that mechanism, or is about something else.",
}


def tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(t) > 1]


class Repo:
    def __init__(self, name: str):
        self.name = name
        self.meta = json.loads((INDEX / f"{name}.chunks.json").read_text(encoding="utf-8"))
        self.vecs = np.load(INDEX / f"{name}.f16.npy").astype(np.float32)
        self.vecs /= np.linalg.norm(self.vecs, axis=1, keepdims=True)
        self._bm25 = None
        self._lines: dict[str, list[str]] = {}

    def lines(self, file: str) -> list[str]:
        if file not in self._lines:
            self._lines[file] = (TARGETS / self.name / file).read_text(encoding="utf-8", errors="replace").splitlines()
        return self._lines[file]

    def text(self, i: int) -> str:
        m = self.meta[i]
        return "\n".join(self.lines(m["file"])[m["start"] - 1:m["end"]])[:MAX_CHARS]

    def bm25(self, query: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        if self._bm25 is None:
            # Inverted index: a query touches only the windows that contain its terms.
            postings: dict[str, tuple[list[int], list[int]]] = {}
            lens = np.zeros(len(self.meta), dtype=np.float32)
            for i, m in enumerate(self.meta):
                counts = Counter(tokens(m["file"] + "\n" + self.text(i)))
                lens[i] = sum(counts.values())
                for t, f in counts.items():
                    ids, tfs = postings.setdefault(t, ([], []))
                    ids.append(i)
                    tfs.append(f)
            norm = (k1 * (1 - b + b * lens / lens.mean())).astype(np.float32)
            self._bm25 = ({t: (np.array(ids), np.array(tfs)) for t, (ids, tfs) in postings.items()}, norm)
        postings, norm = self._bm25
        n = len(norm)
        scores = np.zeros(n, dtype=np.float32)
        for t in set(tokens(query)):
            if t in postings:
                ids, tfs = postings[t]
                idf = math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
                scores[ids] += idf * tfs * (k1 + 1) / (tfs + norm[ids])
        return scores


def files_from(order: list[int], meta: list[dict]) -> list[str]:
    return list(dict.fromkeys(meta[i]["file"] for i in order))


def jev_rerank(client, repo: Repo, query: str, shortlist: list[int]) -> tuple[list[int], dict, int]:
    scores: dict[int, float] = {}
    tokens_in = 0

    def one(batch: list[int]):
        nonlocal tokens_in
        questions = {f"w{j}": Noul(instructions={**RELEVANCE, "judge": f"Judge only windows[{j}]."}, criteria=CRITERIA)
                     for j in range(len(batch))}
        state = {"query": query, "windows": [{"path": repo.meta[i]["file"],
                                              "lines": f"{repo.meta[i]['start']}-{repo.meta[i]['end']}",
                                              "code": repo.text(i)} for i in batch]}
        try:
            r = client.system_one(state=state, questions=questions)
        except Exception:
            return
        for j, i in enumerate(batch):
            a = r.answers.get(f"w{j}")
            if a is not None:
                scores[i] = a.noul
        tokens_in += r.usage.input_tokens

    batches = [shortlist[k:k + BATCH] for k in range(0, len(shortlist), BATCH)]
    with ThreadPoolExecutor(len(batches)) as pool:
        list(pool.map(one, batches))
    # sorted() is stable, so ties and unscored windows keep their embedding order.
    order = sorted(shortlist, key=lambda i: (i not in scores, -scores.get(i, 0.0)))
    return order, scores, tokens_in


def main() -> None:
    load_env()
    client = TypeSafeClient(model="jev-latest", retry=RetryPolicy(max_retries=4), timeout=90)
    repos = {r: Repo(r) for r in ("postgres", "cpython")}
    cs = cases()
    qcache = ROOT / "results" / "query-vectors.json"
    qv = json.loads(qcache.read_text()) if qcache.exists() else {}
    missing = [c[lang] for c in cs for lang in ("en", "fi") if c[lang] not in qv]
    if missing:
        v, _ = embed(missing)
        qv.update({q: vec.tolist() for q, vec in zip(missing, v)})
        qcache.write_text(json.dumps(qv))

    for lang in ("en", "fi"):
        out = {}
        for c in cs:
            repo = repos[c["repo"]]
            q = c[lang]
            vec = np.array(qv[q], dtype=np.float32)
            vec /= np.linalg.norm(vec)
            t = time.perf_counter()
            sims = repo.vecs @ vec
            emb_order = list(np.argsort(-sims)[:200])
            emb_ms = (time.perf_counter() - t) * 1000
            bm_order = list(np.argsort(-repo.bm25(q))[:200])
            t = time.perf_counter()
            rr_order, rr_scores, rr_tokens = jev_rerank(client, repo, q, emb_order[:SHORTLIST])
            rr_ms = (time.perf_counter() - t) * 1000
            rows = {}
            for name, order in (("bm25", bm_order), ("emb", emb_order), ("emb+jev", rr_order + emb_order[SHORTLIST:])):
                rows[name] = {"files": files_from(order, repo.meta)[:20],
                              "windows": [[repo.meta[i]["file"], repo.meta[i]["start"], repo.meta[i]["end"]] for i in order[:120]]}
            rows["emb"]["ms"] = emb_ms
            rows["emb+jev"].update({"ms": rr_ms, "tokens": rr_tokens, "usd": rr_tokens / 1e6 * JEV_USD_PER_M,
                                    "scored": len(rr_scores), "top_noul": max(rr_scores.values(), default=None)})
            out[f"{c['repo']}__{c['name']}"] = rows
            print(f"{lang} {c['repo']}/{c['name']}: rerank {rr_ms:.0f} ms, {rr_tokens} tokens", flush=True)
        (ROOT / "results" / f"rankings-{lang}.json").write_text(json.dumps(out, indent=1, default=int), encoding="utf-8")


if __name__ == "__main__":
    main()
