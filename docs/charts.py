"""Draws the README charts from the committed result files. Standard library only.

    python docs/charts.py

Every bar is read from a results file at run time, never typed in, so a chart
cannot drift from the numbers it claims to show.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Categorical slots 1-3 of the reference palette, validated all-pairs for CVD in
# both modes. Light aqua is under 3:1 on the surface, so every bar carries its value.
STYLE = """
.surface { fill: #fcfcfb; } .title { fill: #0b0b0b; } .ink { fill: #52514e; } .grid { stroke: #e4e3df; }
.s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; } .title { fill: #ffffff; } .ink { fill: #c3c2b7; } .grid { stroke: #383835; }
  .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; }
}
text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bars(path: str, title: str, series: list[str], rows: list[tuple[str, list[float | None]]], note: str) -> None:
    """Horizontal grouped bars on a 0-100% axis: one row per method, one bar per series."""
    width, label_w, right = 760, 250, 56
    plot_w = width - label_w - right
    bar_h, gap, row_gap = 10, 2, 14
    row_h = len(series) * (bar_h + gap) - gap + row_gap
    top = 74
    height = top + len(rows) * row_h + 44
    x = lambda v: label_w + plot_w * v  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'role="img" aria-label="{esc(title)}"><style>{STYLE}</style>',
           f'<rect class="surface" width="{width}" height="{height}" rx="8"/>',
           f'<text class="title" x="20" y="30" font-size="15" font-weight="600">{esc(title)}</text>']
    lx = 20
    for i, name in enumerate(series):  # legend: always present for two or more series
        out.append(f'<rect class="s{i + 1}" x="{lx}" y="44" width="10" height="10" rx="2"/>')
        out.append(f'<text class="ink" x="{lx + 16}" y="53" font-size="12">{esc(name)}</text>')
        lx += 26 + 7.2 * len(name)
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{height - 36}" stroke-width="1"/>')
        out.append(f'<text class="ink" x="{x(t):.1f}" y="{height - 20}" font-size="11" text-anchor="middle">{t:.0%}</text>')
    y = top
    for label, values in rows:
        block = len(series) * (bar_h + gap) - gap
        out.append(f'<text class="title" x="{label_w - 12}" y="{y + block / 2 + 4:.1f}" font-size="12" text-anchor="end">{esc(label)}</text>')
        for i, v in enumerate(values):
            by = y + i * (bar_h + gap)
            if v is None:
                out.append(f'<text class="ink" x="{label_w + 4}" y="{by + 9}" font-size="10">not run</text>')
                continue
            w = max(plot_w * v, 3)
            # Square at the baseline, 4px round at the data end.
            out.append(f'<path class="s{i + 1}" d="M{label_w},{by} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bar_h - 8} '
                       f'a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z"/>')
            out.append(f'<text class="ink" x="{label_w + w + 5:.1f}" y="{by + 9}" font-size="10.5">{v:.1%}</text>')
        y += row_h
    out.append(f'<text class="ink" x="20" y="{height - 4}" font-size="10.5">{esc(note)}</text>')
    out.append("</svg>")
    (ROOT / path).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", path)


def main() -> None:
    def strategy(file: str, name: str, level: str) -> float | None:
        d = json.loads((ROOT / file).read_text(encoding="utf-8"))
        s = next((s for s in d["strategies"] if s["name"] == name), None)
        return s[level]["recallAt"]["1"] if s else None

    rerank = [("no reranking (hybrid)", "hybrid"), ("gpt-4.1-mini, logprobs", "hybrid+llm-4.1-mini"),
              ("gpt-5.6-luna, graded", "hybrid+llm-5.6-luna"), ("Voyage rerank-2.5", "hybrid+voyage"),
              ("Jev, batched", "hybrid+jev-batched")]
    bars("docs/rerank.svg", "Reranking: top-1 recall over identical shortlists",
         ["MuPLeR-fi, 200 queries, document", "Finnish agreements, 84 queries, section"],
         [(label, [strategy("results/mupler/latest.json", name, "doc"),
                   strategy("results/fi-tes/latest.json", name, "section")]) for label, name in rerank],
         "Shortlist ceiling: 97.0% MuPLeR, 70.8% private corpus. Source: results/*/latest.json")

    cs = json.loads((ROOT / "codesearch/results/summary.json").read_text(encoding="utf-8"))
    methods = [("jegrep, no index", "jegrep file recall"), ("BM25", "bm25 file recall@jegrep-N"),
               ("embedding index", "emb file recall@jegrep-N"),
               ("embedding index + Jev rerank", "emb+jev file recall@jegrep-N")]
    bars("docs/codesearch.svg", "Code search: labelled files found, same result budget as jegrep",
         ["English queries", "the same queries in Finnish"],
         [(label, [cs["en"]["means"][key], cs["fi"]["means"][key]]) for label, key in methods],
         "20 Postgres and CPython queries from jegrep's own suite. Source: codesearch/results/summary.json")

    cl = json.loads((ROOT / "classify/results/summary.json").read_text(encoding="utf-8"))
    arms = [("Jev, label names", "jev-zs"), ("gpt-6-luna, label names", "luna6-zs"),
            ("gpt-6-sol, label names", "sol6-zs"), ("Jev, label definitions", "jev-desc"),
            ("Jev, 10 nearest examples", "jev-hybrid"), ("gpt-6-luna, 10 nearest examples", "luna6-fewshot"),
            ("trained: embeddings + LR", "emb-lr")]
    bars("docs/classify.svg", "Intent classification accuracy, 600 messages per dataset",
         ["BANKING77 (77 labels)", "CLINC150 + out-of-scope", "MASSIVE Finnish (60)"],
         [(label, [cl[ds].get(arm, {}).get("acc") for ds in ("banking77", "clinc", "massive-fi")]) for label, arm in arms],
         "95% intervals are about ±3 points. Source: classify/results/summary.json")


if __name__ == "__main__":
    main()
