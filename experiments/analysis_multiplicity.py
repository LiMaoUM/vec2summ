"""Multiplicity correction (Holm) + standardized effect sizes with bootstrap CIs
for the paper's primary comparisons. Reuses the exact per-cell loaders from
analysis_r1_stats.py so conditions match the tables. Writes
results/tables/stats_multiplicity.json.

Run from revision/experiments/.
"""
import json, glob, csv, random
from collections import defaultdict
from statistics import mean, stdev

DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
CELLS = [(d, s) for d in DATASETS for s in SEEDS]
rng = random.Random(2027)


def signflip_p(diffs, n=20000):
    obs = mean(diffs)
    cnt = sum(1 for _ in range(n)
              if abs(mean(d if rng.random() < .5 else -d for d in diffs)) >= abs(obs) - 1e-15)
    return cnt / n


def dz_ci(diffs, B=10000):
    """Paired standardized effect size d_z = mean/sd, with percentile bootstrap CI."""
    d = mean(diffs) / stdev(diffs) if stdev(diffs) > 0 else 0.0
    boot = []
    k = len(diffs)
    for _ in range(B):
        samp = [diffs[rng.randrange(k)] for _ in range(k)]
        sd = stdev(samp)
        boot.append(mean(samp) / sd if sd > 0 else 0.0)
    boot.sort()
    return d, boot[int(.025 * B)], boot[int(.975 * B)]


def holm(pmap):
    """Holm-Bonferroni step-down. pmap: {name: p}. Returns {name: p_adj}."""
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (name, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[name] = round(run, 4)
    return adj


# ---------- recon (Finding 1): e1 + e9 ----------
recon = defaultdict(dict)
for ds in DATASETS:
    for s in SEEDS:
        e1 = json.load(open(f"results/e1/{ds}_1000_{s}.json"))
        for cond in ["vec2summ", "centroid", "random"]:
            recon[cond][(ds, s)] = e1["conditions"][cond]["recon"]["mean_max_sim"]
        for scheme in ["gmm", "pca"]:
            e9 = json.load(open(f"results/e9/{ds}_1000_{s}_{scheme}.json"))
            recon[f"vec2summ_{scheme}"][(ds, s)] = e9["conditions"][f"vec2summ_{scheme}"]["recon"]["mean_max_sim"]


def recon_diffs(a, b):
    return [recon[a][c] - recon[b][c] for c in CELLS]


recon_fam = {"gmm_vs_random": ("vec2summ_gmm", "random"),
             "gmm_vs_centroid": ("vec2summ_gmm", "centroid"),
             "gmm_vs_gauss": ("vec2summ_gmm", "vec2summ"),
             "pca_vs_random": ("vec2summ_pca", "random")}
recon_p, recon_eff = {}, {}
for name, (a, b) in recon_fam.items():
    diffs = recon_diffs(a, b)
    recon_p[name] = signflip_p(diffs)
    d, lo, hi = dz_ci(diffs)
    recon_eff[name] = {"d_z": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)]}
recon_holm = holm(recon_p)

# ---------- GPT-4.1 quality (Finding via Table 2): geval_scores_raw.csv ----------
geval = defaultdict(list)
for r in csv.DictReader(open("results/tables/geval_scores_raw.csv")):
    if r["n"] != "1000" or r["T"]:
        continue
    for c in ["coverage", "factual_accuracy", "coherence", "conciseness"]:
        geval[(r["condition"], r["summarizer"], c)].append(float(r[c]))

gpt_p, gpt_eff = {}, {}
for base in ["random", "centroid"]:
    for c in ["coverage", "factual_accuracy", "coherence", "conciseness"]:
        a = geval[("vec2summ_gmm", "gpt-4.1", c)]
        b = geval[(base, "gpt-4.1", c)]
        if not (a and b):
            continue
        diffs = [x - y for x, y in zip(a, b)]
        key = f"gmm_vs_{base}|{c}"
        gpt_p[key] = signflip_p(diffs)
        if base == "random" and c == "factual_accuracy":
            d, lo, hi = dz_ci(diffs)
            gpt_eff[key] = {"d_z": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)]}
gpt_holm = holm(gpt_p)

# ---------- medoid (Finding 2): per-cell codex + sonnet via keymap ----------
keymap = json.load(open("results/codex_packets/_keymap.json"))


def medoid_cov_diffs(judge_dir):
    diffs = []
    for ds in DATASETS:
        for s in SEEDS:
            cell = f"{ds}_{s}"
            km = {v: k for k, v in keymap[cell].items()}  # cond -> letter
            sc = json.load(open(f"results/{judge_dir}/{cell}.json"))
            diffs.append(sc[km["medoid"]]["coverage"] - sc[km["vec2summ_gmm"]]["coverage"])
    return diffs


med = {}
for judge, d in [("codex", "codex_judge"), ("sonnet", "medoid_sonnet")]:
    diffs = medoid_cov_diffs(d)
    p = signflip_p(diffs)
    dz, lo, hi = dz_ci(diffs)
    med[judge] = {"mean_diff": round(mean(diffs), 3), "p_raw": round(p, 4),
                  "d_z": round(dz, 2), "ci95": [round(lo, 2), round(hi, 2)]}

out = {
    "method": "Holm-Bonferroni step-down within each test family; paired d_z = mean(diff)/sd(diff) with 10k-sample percentile bootstrap 95% CI over the 21 dataset x seed cells.",
    "recon_finding1": {"family_size": len(recon_p),
                       "p_raw": {k: round(v, 4) for k, v in recon_p.items()},
                       "p_holm": recon_holm, "effect": recon_eff},
    "gpt41_quality": {"family_size": len(gpt_p),
                      "p_raw": {k: round(v, 4) for k, v in gpt_p.items()},
                      "p_holm": gpt_holm, "effect_pivotal": gpt_eff},
    "medoid_coverage_finding2": med,
}
json.dump(out, open("results/tables/stats_multiplicity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
