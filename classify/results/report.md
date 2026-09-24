# Results

## banking77 — 77 labels, n=600

| arm | uses train data | acc (95% CI) | macro-F1 | auto-answered at 95% / 98% acc | ECE | p50 ms | $/1000 |
|---|---|---|---|---|---|---|---|
| `tfidf-lr` | yes | **90.5%** (87.9%–92.6%) | 0.900 | 89% / 78% | 0.063 | — | $0.000 |
| `emb-lr` | yes | **94.2%** (92.0%–95.8%) | 0.934 | 98% / 89% | 0.069 | — | $0.002 |
| `emb-knn10` | yes | **93.2%** (90.9%–94.9%) | 0.925 | 96% / 87% | 0.032 | — | $0.002 |
| `emb-zeroshot` | no | **62.5%** (58.6%–66.3%) | 0.608 | 20% / 17% | 0.197 | — | $0.002 |
| `jev-zs` | no | **77.8%** (74.3%–81.0%) | 0.773 | 55% / 34% | 0.104 | 237 | $0.044 |
| `jev-desc` | definitions from 8/label | **87.0%** (84.1%–89.5%) | 0.867 | 72% / 11% | 0.060 | 247 | $0.122 |
| `jev-fewshot` | 10 nearest | **93.0%** (90.7%–94.8%) | 0.924 | 96% / 74% | 0.035 | 237 | $0.059 |
| `jev-hybrid` | 10 nearest | **93.7%** (91.4%–95.4%) | 0.931 | 97% / 77% | 0.033 | 235 | $0.034 |
| `luna-zs` | no | **84.0%** (80.9%–86.7%) | 0.831 | no probability | — | 1996 | $0.113 |
| `luna-desc` | definitions from 8/label | **89.7%** (87.0%–91.9%) | 0.892 | no probability | — | 1980 | $0.407 |
| `luna-fewshot` | 10 nearest | **94.2%** (92.0%–95.8%) | 0.936 | no probability | — | 1852 | $0.159 |
| `mini-zs` | no | **73.5%** (69.8%–76.9%) | 0.732 | 22% / 4% | 0.237 | 2258 | $0.240 |
| `mini-fewshot` | 10 nearest | **91.8%** (89.4%–93.8%) | 0.912 | 92% / 60% | 0.073 | 2220 | $0.331 |
| `sol-zs` | no | **86.3%** (83.4%–88.9%) | 0.854 | no probability | — | 1709 | $1.310 |
| `luna6-zs` | no | **76.8%** (73.3%–80.0%) | 0.764 | 41% / 18% | 0.173 | 1686 | $0.062 |
| `luna6-fewshot` | 10 nearest | **93.5%** (91.2%–95.2%) | 0.929 | 93% / 56% | 0.062 | 1450 | $0.084 |
| `sol6-zs` | no | **82.7%** (79.4%–85.5%) | 0.823 | no probability | — | 1768 | $1.258 |

Paired (McNemar exact): right only in A / right only in B, p

- `jev-zs` vs `sol6-zs`: 13 / 42, p = 0.000114
- `jev-zs` vs `luna6-zs`: 39 / 33, p = 0.556
- `jev-zs` vs `sol-zs`: 8 / 59, p = 1.02e-10
- `jev-zs` vs `luna-zs`: 17 / 54, p = 1.25e-05
- `jev-desc` vs `jev-zs`: 68 / 13, p = 3.81e-10
- `jev-desc` vs `luna-desc`: 12 / 28, p = 0.0166
- `jev-fewshot` vs `luna-fewshot`: 3 / 10, p = 0.0923
- `jev-hybrid` vs `luna6-fewshot`: 8 / 7, p = 1
- `jev-hybrid` vs `emb-lr`: 8 / 11, p = 0.648
- `jev-fewshot` vs `emb-lr`: 7 / 14, p = 0.189

## clinc — 151 labels, n=600

| arm | uses train data | acc (95% CI) | macro-F1 | in-scope acc | OOS recall / precision | auto-answered at 95% / 98% acc | ECE | p50 ms | $/1000 |
|---|---|---|---|---|---|---|---|---|---|
| `tfidf-lr` | yes | **83.3%** (80.1%–86.1%) | 0.857 | 91.0% | 44% / 85% | 78% / 62% | 0.073 | — | $0.000 |
| `emb-lr` | yes | **93.7%** (91.4%–95.4%) | 0.952 | 98.0% | 72% / 96% | 98% / 86% | 0.087 | — | $0.002 |
| `emb-knn10` | yes | **81.5%** (78.2%–84.4%) | 0.869 | 94.2% | 17% / 100% | 75% / 27% | 0.055 | — | $0.002 |
| `emb-zeroshot` | no | **57.0%** (53.0%–60.9%) | 0.570 | 68.3% | 0% / 0% | 25% / 21% | 0.238 | — | $0.002 |
| `jev-zs` | no | **90.5%** (87.9%–92.6%) | 0.898 | 91.0% | 88% / 90% | 87% / 56% | 0.045 | 240 | $0.062 |
| `jev-desc` | definitions from 8/label | **94.2%** (92.0%–95.8%) | 0.954 | 97.2% | 79% / 98% | 98% / 88% | 0.023 | 260 | $0.190 |
| `jev-fewshot` | 10 nearest | **94.7%** (92.6%–96.2%) | 0.967 | 98.8% | 74% / 97% | 100% / 88% | 0.022 | 244 | $0.075 |
| `jev-hybrid` | 10 nearest | **94.8%** (92.8%–96.3%) | 0.965 | 98.6% | 76% / 97% | 100% / 90% | 0.014 | 250 | $0.033 |
| `luna-zs` | no | **91.7%** (89.2%–93.6%) | 0.905 | 93.0% | 85% / 89% | no probability | — | 2127 | $0.150 |
| `luna-desc` | definitions from 8/label | **93.2%** (90.9%–94.9%) · 1 failed | 0.954 | 97.4% | 72% / 96% | no probability | — | 2332 | $0.607 |
| `luna-fewshot` | 10 nearest | **94.5%** (92.4%–96.1%) | 0.965 | 99.0% | 72% / 99% | no probability | — | 2122 | $0.187 |
| `mini-zs` | no | **80.8%** (77.5%–83.8%) | 0.802 | 84.8% | 61% / 91% | 61% / 11% | 0.156 | 1904 | $0.349 |
| `mini-fewshot` | 10 nearest | **90.2%** (87.5%–92.3%) | 0.928 | 96.6% | 58% / 98% | 90% / 52% | 0.080 | 1898 | $0.423 |
| `sol-zs` | no | **94.5%** (92.4%–96.1%) | 0.939 | 94.6% | 94% / 95% | no probability | — | 1966 | $1.612 |
| `luna6-zs` | no | **81.0%** (77.7%–83.9%) | 0.782 | 80.0% | 86% / 70% | 51% / 28% | 0.093 | 1658 | $0.089 |
| `luna6-fewshot` | 10 nearest | **90.2%** (87.5%–92.3%) | 0.883 | 91.0% | 86% / 94% | 90% / 83% | 0.077 | 1468 | $0.107 |
| `sol6-zs` | no | **94.2%** (92.0%–95.8%) | 0.936 | 94.4% | 93% / 90% | no probability | — | 1649 | $1.565 |

Paired (McNemar exact): right only in A / right only in B, p

- `jev-zs` vs `sol6-zs`: 16 / 38, p = 0.00384
- `jev-zs` vs `luna6-zs`: 77 / 20, p = 4.59e-09
- `jev-zs` vs `sol-zs`: 11 / 35, p = 0.000536
- `jev-zs` vs `luna-zs`: 23 / 30, p = 0.41
- `jev-desc` vs `jev-zs`: 35 / 13, p = 0.00209
- `jev-desc` vs `luna-desc`: 18 / 12, p = 0.362
- `jev-fewshot` vs `luna-fewshot`: 11 / 10, p = 1
- `jev-hybrid` vs `luna6-fewshot`: 44 / 16, p = 0.000394
- `jev-hybrid` vs `emb-lr`: 22 / 15, p = 0.324
- `jev-fewshot` vs `emb-lr`: 21 / 15, p = 0.405

## massive-fi — 60 labels, n=600

| arm | uses train data | acc (95% CI) | macro-F1 | auto-answered at 95% / 98% acc | ECE | p50 ms | $/1000 |
|---|---|---|---|---|---|---|---|
| `tfidf-lr` | yes | **83.3%** (80.1%–86.1%) | 0.804 | 76% / 56% | 0.033 | — | $0.000 |
| `emb-lr` | yes | **89.2%** (86.4%–91.4%) | 0.880 | 88% / 79% | 0.062 | — | $0.002 |
| `emb-knn10` | yes | **84.7%** (81.6%–87.3%) | 0.809 | 74% / 63% | 0.037 | — | $0.002 |
| `emb-zeroshot` | no | **48.7%** (44.7%–52.7%) | 0.446 | 7% / 1% | 0.227 | — | $0.002 |
| `jev-zs` | no | **80.3%** (77.0%–83.3%) | 0.817 | 57% / 19% | 0.061 | 240 | $0.034 |
| `jev-desc` | definitions from 8/label | **84.2%** (81.0%–86.9%) | 0.825 | 73% / 25% | 0.072 | 248 | $0.101 |
| `jev-fewshot` | 10 nearest | **89.0%** (86.2%–91.3%) | 0.872 | 87% / 76% | 0.055 | 236 | $0.048 |
| `jev-hybrid` | 10 nearest | **89.0%** (86.2%–91.3%) | 0.856 | 88% / 78% | 0.049 | 239 | $0.032 |
| `luna-zs` | no | **79.5%** (76.1%–82.5%) | 0.794 | no probability | — | 2113 | $0.077 |
| `luna-desc` | definitions from 8/label | **86.2%** (83.2%–88.7%) | 0.859 | no probability | — | 2643 | $0.333 |
| `luna-fewshot` | 10 nearest | **90.2%** (87.5%–92.3%) | 0.870 | no probability | — | 2856 | $0.117 |
| `mini-zs` | no | **74.7%** (71.0%–78.0%) | 0.733 | 12% / 5% | 0.225 | 1893 | $0.166 |
| `mini-fewshot` | 10 nearest | **85.3%** (82.3%–87.9%) | 0.813 | 77% / 40% | 0.132 | 1874 | $0.245 |
| `sol-zs` | no | **82.5%** (79.3%–85.3%) | 0.819 | no probability | — | 1740 | $0.961 |
| `luna6-zs` | no | **71.7%** (67.9%–75.1%) | 0.704 | 54% / 36% | 0.166 | 1495 | $0.043 |
| `luna6-fewshot` | 10 nearest | **85.3%** (82.3%–87.9%) | 0.811 | 81% / 60% | 0.103 | 1436 | $0.063 |
| `sol6-zs` | no | **83.5%** (80.3%–86.3%) | 0.837 | no probability | — | 1668 | $0.913 |

Paired (McNemar exact): right only in A / right only in B, p

- `jev-zs` vs `sol6-zs`: 17 / 36, p = 0.0127
- `jev-zs` vs `luna6-zs`: 80 / 28, p = 5.65e-07
- `jev-zs` vs `sol-zs`: 19 / 32, p = 0.0919
- `jev-zs` vs `luna-zs`: 41 / 36, p = 0.649
- `jev-desc` vs `jev-zs`: 37 / 14, p = 0.00177
- `jev-desc` vs `luna-desc`: 21 / 33, p = 0.134
- `jev-fewshot` vs `luna-fewshot`: 9 / 16, p = 0.23
- `jev-hybrid` vs `luna6-fewshot`: 33 / 11, p = 0.00126
- `jev-hybrid` vs `emb-lr`: 27 / 28, p = 1
- `jev-fewshot` vs `emb-lr`: 26 / 27, p = 1
