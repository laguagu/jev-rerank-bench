"""Classifiers that learn from the training split, plus an embedding zero-shot.

These are the baselines a typed-decision model has to beat: they are free or
nearly free to run, and they use the labelled data the zero-shot arms ignore.
Also writes the nearest training examples each few-shot arm receives.

    uv run python src/local_arms.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from common import DATA, DATASETS, OOS_DESCRIPTION, humanize, load, load_env, result_path
from embed import embed

K_NEIGHBOURS = 10


def write(ds: str, arm: str, test, probs: np.ndarray, classes: list[str], ms: float) -> None:
    with result_path(ds, arm).open("w", encoding="utf-8") as f:
        for ex, p in zip(test, probs):
            order = np.argsort(-p)[:10]
            f.write(json.dumps({
                "id": ex.id, "gold": ex.label, "pred": classes[order[0]], "conf": float(p[order[0]]),
                "top": [[classes[i], round(float(p[i]), 4)] for i in order], "ms": ms, "in_tok": 0, "out_tok": 0,
            }) + "\n")


def run(name: str) -> None:
    d = load(name)
    xtr = [e.text for e in d.train]
    ytr = [e.label for e in d.train]
    xte = [e.text for e in d.test]

    # TF-IDF word + character n-grams: the classic strong baseline for short intents.
    t = time.perf_counter()
    tfidf = make_pipeline(
        FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000)),
                      ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                            min_df=2, max_features=60000))]),
        # saga keeps memory at the size of the weights; lbfgs needs ~10 copies,
        # which does not fit alongside everything else on this machine.
        LogisticRegression(C=20, solver="saga", max_iter=200, tol=1e-3),
    )
    tfidf.fit(xtr, ytr)
    p = tfidf.predict_proba(xte)
    write(name, "tfidf-lr", d.test, p, list(tfidf.classes_), (time.perf_counter() - t) * 1000 / len(xte))

    etr = embed(xtr, f"{name}-train")
    ete = embed(xte, f"{name}-test")

    t = time.perf_counter()
    lr = LogisticRegression(C=10, max_iter=3000)
    lr.fit(etr, ytr)
    p = lr.predict_proba(ete)
    write(name, "emb-lr", d.test, p, list(lr.classes_), (time.perf_counter() - t) * 1000 / len(xte))

    # Zero-shot: cosine to the label text, softmax with a temperature typical
    # for normalised embeddings. Uses no training data.
    label_text = [OOS_DESCRIPTION if l == d.oos_label else humanize(l) for l in d.labels]
    el = embed(label_text, f"{name}-labels")
    sims = ete @ el.T
    z = np.exp((sims - sims.max(axis=1, keepdims=True)) / 0.02)
    write(name, "emb-zeroshot", d.test, z / z.sum(axis=1, keepdims=True), d.labels, 0.0)

    # Nearest training examples for the few-shot arms, plus a kNN vote as its own baseline.
    sim = ete @ etr.T
    nn = np.argsort(-sim, axis=1)[:, :K_NEIGHBOURS]
    classes = sorted(set(ytr))
    idx = {c: i for i, c in enumerate(classes)}
    votes = np.zeros((len(xte), len(classes)))
    for r, row in enumerate(nn):
        for j in row:
            votes[r, idx[ytr[j]]] += sim[r, j]
    write(name, "emb-knn10", d.test, votes / votes.sum(axis=1, keepdims=True), classes, 0.0)

    neighbours = {ex.id: [{"text": xtr[j], "intent": ytr[j]} for j in row] for ex, row in zip(d.test, nn)}
    (DATA / f"neighbours-{name}.json").write_text(json.dumps(neighbours, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: local arms written", flush=True)


if __name__ == "__main__":
    load_env()
    for name in sys.argv[1:] or DATASETS:
        run(name)
