"""R5 cross-judge analysis.
Sonnet = independent second judge (different model family from GPT-4.1).
Opus   = blinded LLM-annotator preference study (NOT human eval).

Un-blind via _keymap.json, then:
 - Sonnet per-condition means (vs GPT-4.1 Table 2) + ranking preservation
 - GPT-4.1 vs Sonnet agreement (Spearman over cells; exact-match rate)
 - Opus pairwise win rates: gmm vs random, gmm vs centroid (faithfulness, coverage)
Outputs: tables/stats_r5.json + tables/stats_r5.md
"""
import json, glob, os
from collections import defaultdict
from statistics import mean, stdev
import csv, math, random

RES = "results"
DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
CRITS = ["coverage", "factual_accuracy", "coherence", "conciseness"]
rng = random.Random(7)
keymap = json.load(open(f"{RES}/judge_packets/_keymap.json"))

# ---- Sonnet scores, un-blinded: sonnet[cond][crit] = list over 21 cells ----
sonnet = defaultdict(lambda: defaultdict(list))
for ds in DATASETS:
    for s in SEEDS:
        cell = f"{ds}_{s}"
        scores = json.load(open(f"{RES}/judge_sonnet/{cell}.json"))
        inv = {lbl: cond for lbl, cond in keymap[cell].items()}
        for lbl, sc in scores.items():
            cond = inv[lbl]
            for c in CRITS:
                sonnet[cond][c].append(sc[c])

CONDS = ["direct", "centroid", "random", "vec2summ_gmm", "vec2summ_pca", "gauss"]
sonnet_means = {cond: {c: (mean(sonnet[cond][c]), stdev(sonnet[cond][c])) for c in CRITS}
                for cond in CONDS}

# ---- GPT-4.1 means (recompute from raw, gpt-4.1, n=1000, T blank) for the same conds ----
gpt = defaultdict(lambda: defaultdict(list))
NAME = {"vec2summ_gmm": "vec2summ_gmm", "vec2summ_pca": "vec2summ_pca", "gauss": "vec2summ",
        "centroid": "centroid", "random": "random", "direct": "direct"}
for r in csv.DictReader(open(f"{RES}/tables/geval_scores_raw.csv")):
    if r["n"] != "1000" or r["T"] or r["summarizer"] != "gpt-4.1":
        continue
    for cond, raw in NAME.items():
        if r["condition"] == raw:
            for c in CRITS:
                gpt[cond][c].append(float(r[c]))
gpt_means = {cond: {c: mean(gpt[cond][c]) for c in CRITS} for cond in CONDS}

# ---- agreement: pair Sonnet & GPT-4.1 per (cond,cell) on coverage+factual ----
def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1):
                rk[order[k]] = avg
            i = j+1
        return rk
    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else 0.0

# build aligned per-cell vectors across all conds×cells for cov & fact
agree = {}
for c in ["coverage", "factual_accuracy"]:
    gx, sx = [], []
    for ci, cond in enumerate(CONDS):
        gx += gpt[cond][c]
        sx += sonnet[cond][c]
    exact = mean(1.0 if abs(a-b) < 1e-9 else 0.0 for a, b in zip(gx, sx))
    within1 = mean(1.0 if abs(a-b) <= 1 else 0.0 for a, b in zip(gx, sx))
    agree[c] = {"spearman": spearman(gx, sx), "exact": exact, "within1": within1,
                "gpt_mean": mean(gx), "sonnet_mean": mean(sx), "n": len(gx)}

# ranking preservation: does each judge rank gmm vs random / centroid the same direction?
def direction(means, a, b, crit):
    return (means[a][crit][0] if isinstance(means[a][crit], tuple) else means[a][crit]) - \
           (means[b][crit][0] if isinstance(means[b][crit], tuple) else means[b][crit])
rank_pres = {}
for crit in ["coverage", "factual_accuracy"]:
    row = {}
    for base in ["random", "centroid"]:
        g = direction(gpt_means, "vec2summ_gmm", base, crit)
        s = direction(sonnet_means, "vec2summ_gmm", base, crit)
        row[base] = {"gpt_sign": (g > 0) - (g < 0), "sonnet_sign": (s > 0) - (s < 0),
                     "gpt_diff": g, "sonnet_diff": s,
                     "agree": (g > 0) == (s > 0) or (g == 0 and s == 0)}
    rank_pres[crit] = row

# ---- Opus pairwise win rates from rankings ----
def winrate(dim):
    wins = defaultdict(lambda: [0, 0])   # base -> [gmm_better, total]
    for ds in DATASETS:
        for s in SEEDS:
            cell = f"{ds}_{s}"
            rk = json.load(open(f"{RES}/judge_opus/{cell}.json"))[dim]
            inv = keymap[cell]
            lbl2cond = inv
            pos = {lbl2cond[lbl]: i for i, lbl in enumerate(rk)}  # lower index = better
            for base in ["random", "centroid"]:
                if "vec2summ_gmm" in pos and base in pos:
                    wins[base][1] += 1
                    if pos["vec2summ_gmm"] < pos[base]:
                        wins[base][0] += 1
    return {b: {"gmm_wins": w[0], "n": w[1], "rate": w[0]/w[1]} for b, w in wins.items()}

def sign_p(k, n):
    from math import comb
    kk = min(k, n-k)
    return min(1.0, 2*sum(comb(n, i) for i in range(kk+1))/2**n)

opus = {}
for dim in ["faithfulness_rank", "coverage_rank"]:
    wr = winrate(dim)
    for b in wr:
        wr[b]["sign_p"] = sign_p(wr[b]["gmm_wins"], wr[b]["n"])
    opus[dim] = wr

# also Opus mean rank position per condition (1=best..6=worst)
opus_meanrank = {}
for dim in ["faithfulness_rank", "coverage_rank"]:
    pos = defaultdict(list)
    for ds in DATASETS:
        for s in SEEDS:
            cell = f"{ds}_{s}"
            rk = json.load(open(f"{RES}/judge_opus/{cell}.json"))[dim]
            inv = keymap[cell]
            for i, lbl in enumerate(rk):
                pos[inv[lbl]].append(i+1)
    opus_meanrank[dim] = {c: mean(pos[c]) for c in CONDS}

out = {"sonnet_means": {c: {k: {"mean": v[0], "sd": v[1]} for k, v in d.items()}
                        for c, d in sonnet_means.items()},
       "gpt_means": gpt_means, "agreement": agree,
       "ranking_preservation": rank_pres,
       "opus_pairwise": opus, "opus_mean_rank": opus_meanrank}
json.dump(out, open(f"{RES}/tables/stats_r5.json", "w"), indent=1)

L = ["# R5 cross-judge analysis\n",
     "Sonnet = independent 2nd judge (Claude, different family from GPT-4.1).",
     "Opus = blinded LLM-annotator preference study (NOT human evaluation).\n",
     "## Sonnet vs GPT-4.1 per-condition means (21 cells)\n",
     "| cond | cov G/S | fact G/S | coher G/S | concis G/S |",
     "|---|---|---|---|---|"]
for cond in CONDS:
    L.append(f"| {cond} | "
             + " | ".join(f"{gpt_means[cond][c]:.2f}/{sonnet_means[cond][c][0]:.2f}" for c in CRITS)
             + " |")
L.append("\n## Judge agreement (GPT-4.1 vs Sonnet, per-cell)\n")
for c, a in agree.items():
    L.append(f"- {c}: Spearman {a['spearman']:.2f}, exact-match {a['exact']*100:.0f}%, "
             f"within-1 {a['within1']*100:.0f}%, means G={a['gpt_mean']:.2f}/S={a['sonnet_mean']:.2f} (n={a['n']})")
L.append("\n## Ranking preservation (sign of gmm-minus-baseline diff)\n")
for crit, row in rank_pres.items():
    for base, d in row.items():
        L.append(f"- {crit}, gmm vs {base}: GPT diff {d['gpt_diff']:+.2f}, "
                 f"Sonnet diff {d['sonnet_diff']:+.2f} -> {'AGREE' if d['agree'] else 'DISAGREE'}")
L.append("\n## Opus blinded preference: gmm win rate (21 cells)\n")
for dim, wr in opus.items():
    for base, w in wr.items():
        L.append(f"- {dim} gmm vs {base}: {w['gmm_wins']}/{w['n']} = {w['rate']*100:.0f}% "
                 f"(sign p={w['sign_p']:.3f})")
L.append("\n## Opus mean rank position (1=best of 6 .. 6=worst)\n")
for dim, mr in opus_meanrank.items():
    L.append(f"- {dim}: " + ", ".join(f"{c} {mr[c]:.2f}" for c in CONDS))
open(f"{RES}/tables/stats_r5.md", "w").write("\n".join(L)+"\n")
print("\n".join(L))
