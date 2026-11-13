# R1 statistics (computed from per-seed artifacts)

## Table 1 recon: per-dataset mean ± SD (3 seeds)

| dataset | vec2summ_gmm | random | centroid | vec2summ_pca | vec2summ |
|---|---|---|---|---|---|
| askdocs | 0.813±0.001 | 0.788±0.008 | 0.809±0.003 | 0.803±0.002 | 0.775±0.002 |
| askdocs_fever | 0.847±0.001 | 0.827±0.001 | 0.840±0.000 | 0.843±0.002 | 0.807±0.004 |
| biden | 0.853±0.002 | 0.838±0.007 | 0.847±0.001 | 0.845±0.004 | 0.810±0.003 |
| illegal | 0.869±0.003 | 0.849±0.003 | 0.862±0.001 | 0.863±0.002 | 0.830±0.005 |
| onebook | 0.906±0.001 | 0.897±0.003 | 0.905±0.002 | 0.904±0.001 | 0.867±0.009 |
| tifu | 0.821±0.004 | 0.811±0.002 | 0.828±0.003 | 0.813±0.001 | 0.793±0.006 |
| trump | 0.856±0.002 | 0.833±0.003 | 0.852±0.000 | 0.850±0.003 | 0.812±0.015 |

## Paired tests (recon)

- gmm_vs_random: mean diff +0.0173 (SD 0.0068), sign-flip p=0.0000 (21 cells), dataset wins 7/7, sign test p=0.0156
- gmm_vs_centroid: mean diff +0.0030 (SD 0.0050), sign-flip p=0.0140 (21 cells), dataset wins 6/7, sign test p=0.1250
- gmm_vs_gauss: mean diff +0.0389 (SD 0.0076), sign-flip p=0.0000 (21 cells), dataset wins 7/7, sign test p=0.0156
- pca_vs_random: mean diff +0.0113 (SD 0.0068), sign-flip p=0.0000 (21 cells), dataset wins 7/7, sign test p=0.0156
- gauss_vs_random: mean diff -0.0215 (SD 0.0092), sign-flip p=0.0000 (21 cells), dataset wins 0/7, sign test p=0.0156

## Judge scores: mean ± SD (21 cells) and paired tests

### gpt-4.1
- direct: coverage: 5.00±0.00, factual_accuracy: 5.00±0.00, coherence: 5.00±0.00, conciseness: 3.29±0.46
- centroid: coverage: 3.81±1.03, factual_accuracy: 4.81±0.40, coherence: 5.00±0.00, conciseness: 3.19±0.68
- random: coverage: 3.76±1.18, factual_accuracy: 4.05±1.24, coherence: 5.00±0.00, conciseness: 3.38±0.50
- vec2summ_gmm: coverage: 3.67±1.15, factual_accuracy: 4.71±0.46, coherence: 5.00±0.00, conciseness: 3.05±0.59
- vec2summ_pca: coverage: 3.38±1.28, factual_accuracy: 4.48±0.98, coherence: 4.95±0.22, conciseness: 3.00±0.55
- vec2summ: coverage: 2.38±0.97, factual_accuracy: 3.24±1.22, coherence: 4.95±0.22, conciseness: 2.95±0.38
  - gmm_vs_random|gpt-4.1|coverage: diff -0.10, p=0.8224
  - gmm_vs_random|gpt-4.1|factual_accuracy: diff +0.67, p=0.0310
  - gmm_vs_random|gpt-4.1|coherence: diff +0.00, p=1.0000
  - gmm_vs_random|gpt-4.1|conciseness: diff -0.33, p=0.0906
  - gmm_vs_centroid|gpt-4.1|coverage: diff -0.14, p=0.6450
  - gmm_vs_centroid|gpt-4.1|factual_accuracy: diff -0.10, p=0.6849
  - gmm_vs_centroid|gpt-4.1|coherence: diff +0.00, p=1.0000
  - gmm_vs_centroid|gpt-4.1|conciseness: diff -0.14, p=0.6351
### llama-3.3-70b
- direct: coverage: 3.86±0.96, factual_accuracy: 4.76±0.54, coherence: 4.86±0.36, conciseness: 2.24±0.44
- centroid: coverage: 3.14±0.96, factual_accuracy: 4.81±0.40, coherence: 5.00±0.00, conciseness: 2.38±0.50
- random: coverage: 3.05±1.02, factual_accuracy: 4.24±1.18, coherence: 4.90±0.30, conciseness: 2.52±0.51
- vec2summ_gmm: coverage: 3.00±0.95, factual_accuracy: 4.67±0.48, coherence: 4.90±0.30, conciseness: 2.57±0.60
- vec2summ_pca: coverage: 2.62±0.86, factual_accuracy: 3.95±1.28, coherence: 5.00±0.00, conciseness: 2.48±0.51
- vec2summ: coverage: 1.86±0.48, factual_accuracy: 3.14±1.42, coherence: 4.90±0.30, conciseness: 2.29±0.46
  - gmm_vs_random|llama-3.3-70b|coverage: diff -0.05, p=1.0000
  - gmm_vs_random|llama-3.3-70b|factual_accuracy: diff +0.43, p=0.2033
  - gmm_vs_random|llama-3.3-70b|coherence: diff +0.00, p=1.0000
  - gmm_vs_random|llama-3.3-70b|conciseness: diff +0.05, p=1.0000
  - gmm_vs_centroid|llama-3.3-70b|coverage: diff -0.14, p=0.6670
  - gmm_vs_centroid|llama-3.3-70b|factual_accuracy: diff -0.14, p=0.5070
  - gmm_vs_centroid|llama-3.3-70b|coherence: diff -0.10, p=0.5044
  - gmm_vs_centroid|llama-3.3-70b|conciseness: diff +0.19, p=0.4836

## Factuality: unsupported claims with Wilson 95% CI

- centroid: 0/252 = 0.0% [-0.0, 1.5]
- direct: 1/251 = 0.4% [0.1, 2.2]
- random: 6/252 = 2.4% [1.1, 5.1]
- vec2summ: 10/252 = 4.0% [2.2, 7.1]
- vec2summ_gmm: 9/252 = 3.6% [1.9, 6.6]
- vec2summ_pca: 6/250 = 2.4% [1.1, 5.1]
- gmm_vs_random: z=0.79, p=0.4316
- gmm_vs_centroid: z=3.03, p=0.0025
- gmm_vs_direct: z=2.55, p=0.0108
- gmm_vs_vec2summ: z=-0.23, p=0.8151
