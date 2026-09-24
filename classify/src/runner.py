"""Resumable, bounded-concurrency execution of one arm over one dataset sample.

Rows are appended as they finish, so an interrupted run continues where it
stopped, and a failed call is recorded as an error row rather than dropped.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from common import Dataset, Example, done_ids, result_path


def run_arm(d: Dataset, arm: str, fn: Callable[[Example], dict], workers: int = 6) -> None:
    path = result_path(d.name, arm)
    done = done_ids(path)
    todo = [e for e in d.test if e.id not in done]
    if not todo:
        print(f"{d.name}/{arm}: complete", flush=True)
        return
    lock = threading.Lock()
    counter = {"n": 0, "err": 0}
    started = time.perf_counter()

    def one(ex: Example) -> None:
        t = time.perf_counter()
        try:
            row = fn(ex)
        except Exception as e:  # recorded, not raised: the rest of the sample still runs
            row = {"error": type(e).__name__}
        row.update({"id": ex.id, "gold": ex.label, "ms": round((time.perf_counter() - t) * 1000)})
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counter["n"] += 1
            counter["err"] += 1 if row.get("error") else 0
            if counter["n"] % 100 == 0:
                print(f"{d.name}/{arm}: {counter['n']}/{len(todo)} errors={counter['err']}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    print(f"{d.name}/{arm}: {counter['n']} rows, {counter['err']} errors, "
          f"{time.perf_counter() - started:.0f}s", flush=True)
