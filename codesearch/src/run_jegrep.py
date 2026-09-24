"""Runs the released jegrep binary on every labelled query, in English and Finnish.

    uv run python src/run_jegrep.py

Query-only: the annotated keywords are never passed (jegrep's own runner calls
that the oracle-assisted track). Default strategy (cascade), default thresholds,
TypeSafe endpoint, jev-latest. One fresh process per query; raw JSON is kept.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from index import ROOT, TARGETS, load_env

BIN = ROOT / "bin" / "jegrep-v0.1.2-x86_64-pc-windows-msvc" / "jegrep.exe"
OUT = ROOT / "results" / "jegrep"


def cases() -> list[dict]:
    fi = json.loads((ROOT / "suites" / "fi-queries.json").read_text(encoding="utf-8"))
    out = []
    for repo in ("postgres", "cpython"):
        for p in sorted((ROOT / "suites" / repo).glob("query_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            name = p.stem[len("query_"):]
            out.append({"repo": repo, "name": name, "en": d["query"], "fi": fi[name], "matches": d["matches"]})
    return out


def main() -> None:
    load_env()  # the child process inherits TYPESAFE_API_KEY from os.environ
    suite = cases()
    for lang in ("en", "fi"):
        (OUT / lang).mkdir(parents=True, exist_ok=True)
        for c in suite:
            path = OUT / lang / f"{c['repo']}__{c['name']}.json"
            if path.exists():
                continue
            t = time.perf_counter()
            r = subprocess.run([str(BIN), c[lang], str(TARGETS / c["repo"]), "--endpoint", "typesafe", "--json", "-q"],
                               capture_output=True, timeout=240)
            wall = time.perf_counter() - t
            try:
                data = json.loads(r.stdout.decode("utf-8"))
            except json.JSONDecodeError:
                data = {"error": f"exit {r.returncode}", "hits": []}
            data["wall_s"] = round(wall, 2)
            data["root"] = f"targets/{c['repo']}"  # jegrep reports this machine's absolute path
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"{lang} {c['repo']}/{c['name']}: {len(data.get('hits', []))} hits, {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
