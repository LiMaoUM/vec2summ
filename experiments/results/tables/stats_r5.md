# R5 cross-judge analysis

Sonnet = independent 2nd judge (Claude, different family from GPT-4.1).
Opus = blinded LLM-annotator preference study (NOT human evaluation).

## Sonnet vs GPT-4.1 per-condition means (21 cells)

| cond | cov G/S | fact G/S | coher G/S | concis G/S |
|---|---|---|---|---|
| direct | 5.00/4.86 | 5.00/4.67 | 5.00/4.95 | 3.29/3.67 |
| centroid | 3.81/3.05 | 4.81/3.95 | 5.00/4.29 | 3.19/3.81 |
| random | 3.76/3.24 | 4.05/3.19 | 5.00/4.14 | 3.38/3.67 |
| vec2summ_gmm | 3.67/3.10 | 4.71/3.29 | 5.00/4.14 | 3.05/3.48 |
| vec2summ_pca | 3.38/2.67 | 4.48/3.10 | 4.95/3.81 | 3.00/3.38 |
| gauss | 2.38/1.95 | 3.24/2.00 | 4.95/3.29 | 2.95/3.05 |

## Judge agreement (GPT-4.1 vs Sonnet, per-cell)

- coverage: Spearman 0.74, exact-match 46%, within-1 86%, means G=3.67/S=3.14 (n=126)
- factual_accuracy: Spearman 0.76, exact-match 27%, within-1 73%, means G=4.38/S=3.37 (n=126)

## Ranking preservation (sign of gmm-minus-baseline diff)

- coverage, gmm vs random: GPT diff -0.10, Sonnet diff -0.14 -> AGREE
- coverage, gmm vs centroid: GPT diff -0.14, Sonnet diff +0.05 -> DISAGREE
- factual_accuracy, gmm vs random: GPT diff +0.67, Sonnet diff +0.10 -> AGREE
- factual_accuracy, gmm vs centroid: GPT diff -0.10, Sonnet diff -0.67 -> AGREE

## Opus blinded preference: gmm win rate (21 cells)

- faithfulness_rank gmm vs random: 8/21 = 38% (sign p=0.383)
- faithfulness_rank gmm vs centroid: 7/21 = 33% (sign p=0.189)
- coverage_rank gmm vs random: 7/21 = 33% (sign p=0.189)
- coverage_rank gmm vs centroid: 10/21 = 48% (sign p=1.000)

## Opus mean rank position (1=best of 6 .. 6=worst)

- faithfulness_rank: direct 1.38, centroid 3.05, random 3.05, vec2summ_gmm 3.76, vec2summ_pca 4.05, gauss 5.71
- coverage_rank: direct 1.00, centroid 3.57, random 2.95, vec2summ_gmm 3.52, vec2summ_pca 4.14, gauss 5.81
