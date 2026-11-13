"""R1 statistics for the EACL revision: SDs, paired tests, Wilson CIs.

Sources: e1 (gauss/centroid/random recon per seed), e9 (gmm/pca recon per seed),
tables/geval_scores_raw.csv (judge scores per cell), e8 (claim-level audits).
Outputs: tables/stats_r1.json + tables/stats_r1.md
"""
import json, glob, csv, math, random
from collections import defaultdict
from statistics import mean, stdev

DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
rng = random.Random(12345)

# ---------- Table 1: reconstruction (mean-max cosine) ----------
recon = defaultdict(dict)  # recon[cond][(ds,seed)] = value
for ds in DATASETS:
    for s in SEEDS:
        e1 = json.load(open(f"e1/{ds}_1000_{s}.json"))
        for cond in ["vec2summ", "centroid", "random"]:
            recon[cond][(ds, s)] = e1["conditions"][cond]["recon"]["mean_max_sim"]
        for scheme in ["gmm", "pca"]:
            e9 = json.load(open(f"e9/{ds}_1000_{s}_{scheme}.json"))
            key = f"vec2summ_{scheme}"
            recon[key][(ds, s)] = e9["conditions"][key]["recon"]["mean_max_sim"]

def per_dataset(cond):
    out = {}
    for ds in DATASETS:
        vals = [recon[cond][(ds, s)] for s in SEEDS]
        out[ds] = {"mean": mean(vals), "sd": stdev(vals)}
    return out

def signflip_p(diffs, n=20000):
    """Two-sided sign-flip permutation test on paired differences."""
    obs = mean(diffs)
    cnt = 0
    for _ in range(n):
        m = mean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(m) >= abs(obs) - 1e-15:
            cnt += 1
    return cnt / n

def sign_test_p(wins, total):
    """Two-sided exact binomial sign test, p=0.5."""
    from math import comb
    k = min(wins, total - wins)
    p = sum(comb(total, i) for i in range(k + 1)) / 2 ** total
    return min(1.0, 2 * p)

def paired(condA, condB):
    cells = [(ds, s) for ds in DATASETS for s in SEEDS]
    diffs = [recon[condA][c] - recon[condB][c] for c in cells]
    ds_diffs = {ds: mean(recon[condA][(ds, s)] - recon[condB][(ds, s)] for s in SEEDS) for ds in DATASETS}
    wins = sum(1 for d in ds_diffs.values() if d > 0)
    return {
        "mean_diff": mean(diffs),
        "sd_diff": stdev(diffs),
        "p_signflip_21cells": signflip_p(diffs),
        "dataset_wins": wins,
        "p_signtest_7datasets": sign_test_p(wins, len(DATASETS)),
        "per_dataset_diff": ds_diffs,
    }

table1 = {cond: per_dataset(cond) for cond in recon}
tests1 = {
    "gmm_vs_random": paired("vec2summ_gmm", "random"),
    "gmm_vs_centroid": paired("vec2summ_gmm", "centroid"),
    "gmm_vs_gauss": paired("vec2summ_gmm", "vec2summ"),
    "pca_vs_random": paired("vec2summ_pca", "random"),
    "gauss_vs_random": paired("vec2summ", "random"),
}

# ---------- Table 2: judge scores ----------
geval = defaultdict(list)  # geval[(cond,summ,crit)] = list of 21 cell scores
rows = list(csv.DictReader(open("tables/geval_scores_raw.csv")))
CRITS = ["coverage", "factual_accuracy", "coherence", "conciseness"]
for r in rows:
    if r["n"] != "1000" or r["T"]:
        continue
    for c in CRITS:
        geval[(r["condition"], r["summarizer"], c)].append(float(r[c]))

table2, tests2 = {}, {}
for cond in ["direct", "centroid", "random", "vec2summ_gmm", "vec2summ_pca", "vec2summ"]:
    for summ in ["gpt-4.1", "llama-3.3-70b"]:
        for c in CRITS:
            v = geval[(cond, summ, c)]
            if v:
                table2[f"{cond}|{summ}|{c}"] = {"n": len(v), "mean": mean(v), "sd": stdev(v)}
for summ in ["gpt-4.1", "llama-3.3-70b"]:
    for base in ["random", "centroid"]:
        for c in CRITS:
            a, b = geval[("vec2summ_gmm", summ, c)], geval[(base, summ, c)]
            diffs = [x - y for x, y in zip(a, b)]
            tests2[f"gmm_vs_{base}|{summ}|{c}"] = {
                "mean_diff": mean(diffs), "p_signflip": signflip_p(diffs)}

# ---------- Table 3: factuality (claim counts, Wilson CIs, 2-prop z) ----------
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((ctr - rad) / den, (ctr + rad) / den)

fact = defaultdict(lambda: [0, 0])  # cond -> [unsupported, total]
for f in glob.glob("e8/*.json"):
    if "_100_" in f:  # the n=100 grid is excluded: every other table uses the 21 n=1000 cells
        continue
    d = json.load(open(f))
    for cond, audit in d["audits"].items():
        for cl in audit["claims"]:
            fact[cond][1] += 1
            if cl["verdict"].upper() == "UNSUPPORTED":
                fact[cond][0] += 1

table3 = {}
for cond, (k, n) in fact.items():
    lo, hi = wilson(k, n)
    table3[cond] = {"unsupported": k, "total": n, "rate": k / n,
                    "wilson95": [lo, hi]}

def two_prop_z(k1, n1, k2, n2):
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pv

tests3 = {}
for base in ["random", "centroid", "direct", "vec2summ"]:
    if "vec2summ_gmm" in fact and base in fact:
        z, pv = two_prop_z(*fact["vec2summ_gmm"], *fact[base])
        tests3[f"gmm_vs_{base}"] = {"z": z, "p": pv}

out = {"table1_recon": table1, "tests1": tests1,
       "table2_geval": table2, "tests2": tests2,
       "table3_fact": table3, "tests3": tests3}
json.dump(out, open("tables/stats_r1.json", "w"), indent=1)

# ---------- Markdown summary ----------
L = ["# R1 statistics (computed from per-seed artifacts)\n"]
L.append("## Table 1 recon: per-dataset mean ± SD (3 seeds)\n")
conds = ["vec2summ_gmm", "random", "centroid", "vec2summ_pca", "vec2summ"]
L.append("| dataset | " + " | ".join(conds) + " |")
L.append("|" + "---|" * (len(conds) + 1))
for ds in DATASETS:
    L.append(f"| {ds} | " + " | ".join(
        f"{table1[c][ds]['mean']:.3f}±{table1[c][ds]['sd']:.3f}" for c in conds) + " |")
L.append("\n## Paired tests (recon)\n")
for k, t in tests1.items():
    L.append(f"- {k}: mean diff {t['mean_diff']:+.4f} (SD {t['sd_diff']:.4f}), "
             f"sign-flip p={t['p_signflip_21cells']:.4f} (21 cells), "
             f"dataset wins {t['dataset_wins']}/7, sign test p={t['p_signtest_7datasets']:.4f}")
L.append("\n## Judge scores: mean ± SD (21 cells) and paired tests\n")
for summ in ["gpt-4.1", "llama-3.3-70b"]:
    L.append(f"### {summ}")
    for cond in ["direct", "centroid", "random", "vec2summ_gmm", "vec2summ_pca", "vec2summ"]:
        cells = [f"{c}: {table2[f'{cond}|{summ}|{c}']['mean']:.2f}±{table2[f'{cond}|{summ}|{c}']['sd']:.2f}"
                 for c in CRITS if f"{cond}|{summ}|{c}" in table2]
        if cells:
            L.append(f"- {cond}: " + ", ".join(cells))
    for k, t in tests2.items():
        if summ in k:
            L.append(f"  - {k}: diff {t['mean_diff']:+.2f}, p={t['p_signflip']:.4f}")
L.append("\n## Factuality: unsupported claims with Wilson 95% CI\n")
for cond, t in sorted(table3.items()):
    L.append(f"- {cond}: {t['unsupported']}/{t['total']} = {t['rate']*100:.1f}% "
             f"[{t['wilson95'][0]*100:.1f}, {t['wilson95'][1]*100:.1f}]")
for k, t in tests3.items():
    L.append(f"- {k}: z={t['z']:.2f}, p={t['p']:.4f}")
open("tables/stats_r1.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
