"""Scores jegrep and the index-based rankers against jegrep's own labels.

    uv run python src/score.py

- file recall: gold files found / gold files, macro-averaged over queries
- jegrep returns a set, so it is scored on that set; a ranker is scored on its
  top 5 and top 10 files, and on its top N where N is jegrep's hit count for the
  same query (matched budget, at least 1)
- region recall: annotated regions touched by a returned range (jegrep: its top
  three ranges per hit, as jegrep's own runner scores; rankers: their best windows
  up to the same number of lines jegrep returned for that query, so a whole-file
  range cannot win on size alone)
- labels are positive-only, so extra files are unjudged, not wrong
"""

from __future__ import annotations

import json
import statistics

from index import JEV_USD_PER_M, ROOT
from run_jegrep import cases


def region_recall(ranges: list[tuple[str, int, int]], regions: list[tuple[str, int, int]]) -> float:
    """Share of annotated regions that some returned range overlaps."""
    return sum(any(f == g and a <= e and s <= b for f, a, b in ranges) for g, s, e in regions) / len(regions)


def main() -> None:
    cs = cases()
    lines = ["# Code search results", ""]
    summary = {}
    for lang in ("en", "fi"):
        rankings = json.loads((ROOT / "results" / f"rankings-{lang}.json").read_text(encoding="utf-8"))
        agg: dict[str, list] = {}
        per_query = []
        for c in cs:
            key = f"{c['repo']}__{c['name']}"
            gold_files = {m["file"] for m in c["matches"]}
            regions = [(m["file"], m["start_line"], m["end_line"]) for m in c["matches"]]
            j = json.loads((ROOT / "results" / "jegrep" / lang / f"{key}.json").read_text(encoding="utf-8"))
            hits = [{**h, "path": h["path"].replace("\\", "/")} for h in j.get("hits", [])]
            hit_files = {h["path"] for h in hits}
            ranges = [(h["path"], r["start"], r["end"])
                      for h in hits for r in sorted(h.get("ranges", []), key=lambda r: -(r.get("p") or 0))[:3]]
            n = max(1, len(hits))
            covered = set()
            for f, a, b in ranges:
                covered.update((f, x) for x in range(a, b + 1))
            budget = len(covered)
            stats = j.get("stats", {})
            row = {"query": key, "gold": len(gold_files)}

            def add(name, val):
                agg.setdefault(name, []).append(val)
                row[name] = val

            add("jegrep file recall", len(gold_files & hit_files) / len(gold_files))
            add("jegrep region recall", region_recall(ranges, regions))
            add("jegrep hits", len(hits))
            add("jegrep lines", budget)
            add("jegrep usd", stats.get("input_tokens", 0) / 1e6 * JEV_USD_PER_M)
            add("jegrep s", j.get("wall_s", 0))
            for name in ("bm25", "emb", "emb+jev"):
                files = rankings[key][name]["files"]
                wins = rankings[key][name]["windows"]
                for k in (1, 5, 10):
                    add(f"{name} file recall@{k}", len(gold_files & set(files[:k])) / len(gold_files))
                add(f"{name} file recall@jegrep-N", len(gold_files & set(files[:n])) / len(gold_files))
                taken, used = [], 0
                for f, a, b in wins:
                    if used + (b - a + 1) > max(budget, 80):
                        break
                    taken.append((f, a, b))
                    used += b - a + 1
                add(f"{name} region recall@lines", region_recall(taken, regions))
            rr = rankings[key]["emb+jev"]
            add("emb+jev usd", rr.get("usd", 0))
            add("emb+jev s", rr.get("ms", 0) / 1000)
            per_query.append(row)
        means = {k: statistics.mean(v) for k, v in agg.items()}
        summary[lang] = {"means": means, "per_query": per_query}
        lines += [f"## Queries in {'English' if lang == 'en' else 'Finnish'} (n={len(cs)})", "",
                  "| method | file recall | region recall | files returned | lines returned (mean) | $/query | s/query |", "|---|---|---|---|---|---|---|",
                  f"| jegrep 0.1.2 (cascade, no index) | {means['jegrep file recall']:.0%} | {means['jegrep region recall']:.0%} "
                  f"| {means['jegrep hits']:.1f} | {means['jegrep lines']:.0f} | ${means['jegrep usd']:.4f} | {means['jegrep s']:.1f} |"]
        for name, label in (("bm25", "BM25 over 80-line windows"), ("emb", "embedding index"),
                            ("emb+jev", "embedding index + Jev rerank of top 30")):
            cost, secs = ((f"${means['emb+jev usd']:.4f}", f"{means['emb+jev s']:.1f}") if name == "emb+jev"
                          else ("≈$0", "<0.1"))
            lines.append(f"| {label}, top N = jegrep's count | {means[f'{name} file recall@jegrep-N']:.0%} "
                         f"| {means[f'{name} region recall@lines']:.0%} | same | same | {cost} | {secs} |")
        lines += ["", "| ranker | recall@1 | recall@5 | recall@10 |", "|---|---|---|---|"]
        for name in ("bm25", "emb", "emb+jev"):
            lines.append(f"| {name} | {means[f'{name} file recall@1']:.0%} | {means[f'{name} file recall@5']:.0%} "
                         f"| {means[f'{name} file recall@10']:.0%} |")
        zero = [r["query"] for r in per_query if r["jegrep hits"] == 0]
        lines += ["", f"jegrep returned nothing for {len(zero)} of {len(cs)} queries" + (f": {', '.join(zero)}" if zero else "."), ""]
    (ROOT / "results" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
