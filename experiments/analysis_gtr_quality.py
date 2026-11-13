"""Unblind the GTR scale-quality judging and test whether the document-free gap
widens at n=10k. Same GTR pipeline at both sizes; judges = codex (GPT-5) + Sonnet.
"""
import json, numpy as np
from pathlib import Path

CODEX = json.loads('{"trump_1000_42":{"A":{"coverage":5,"factual_accuracy":5},"B":{"coverage":4,"factual_accuracy":4},"C":{"coverage":4,"factual_accuracy":4},"D":{"coverage":4,"factual_accuracy":5}},"trump_10000_42":{"A":{"coverage":4,"factual_accuracy":4},"B":{"coverage":4,"factual_accuracy":5},"C":{"coverage":3,"factual_accuracy":4},"D":{"coverage":5,"factual_accuracy":5}},"trump_1000_43":{"A":{"coverage":5,"factual_accuracy":5},"B":{"coverage":5,"factual_accuracy":5},"C":{"coverage":4,"factual_accuracy":5},"D":{"coverage":4,"factual_accuracy":4}},"trump_10000_43":{"A":{"coverage":3,"factual_accuracy":3},"B":{"coverage":5,"factual_accuracy":5},"C":{"coverage":4,"factual_accuracy":5},"D":{"coverage":5,"factual_accuracy":5}},"trump_1000_44":{"A":{"coverage":5,"factual_accuracy":4},"B":{"coverage":3,"factual_accuracy":2},"C":{"coverage":5,"factual_accuracy":5},"D":{"coverage":4,"factual_accuracy":5}},"trump_10000_44":{"A":{"coverage":4,"factual_accuracy":5},"B":{"coverage":2,"factual_accuracy":2},"C":{"coverage":4,"factual_accuracy":4},"D":{"coverage":5,"factual_accuracy":5}}}')
SONNET = json.loads('{"trump_1000_42":{"A":{"coverage":4,"factual_accuracy":5},"B":{"coverage":4,"factual_accuracy":5},"C":{"coverage":4,"factual_accuracy":4},"D":{"coverage":4,"factual_accuracy":5}},"trump_10000_42":{"A":{"coverage":4,"factual_accuracy":4},"B":{"coverage":4,"factual_accuracy":5},"C":{"coverage":3,"factual_accuracy":3},"D":{"coverage":4,"factual_accuracy":5}},"trump_1000_43":{"A":{"coverage":4,"factual_accuracy":4},"B":{"coverage":4,"factual_accuracy":5},"C":{"coverage":3,"factual_accuracy":4},"D":{"coverage":4,"factual_accuracy":4}},"trump_10000_43":{"A":{"coverage":3,"factual_accuracy":4},"B":{"coverage":4,"factual_accuracy":5},"C":{"coverage":4,"factual_accuracy":4},"D":{"coverage":4,"factual_accuracy":4}},"trump_1000_44":{"A":{"coverage":4,"factual_accuracy":4},"B":{"coverage":3,"factual_accuracy":2},"C":{"coverage":4,"factual_accuracy":5},"D":{"coverage":4,"factual_accuracy":4}},"trump_10000_44":{"A":{"coverage":4,"factual_accuracy":4},"B":{"coverage":3,"factual_accuracy":2},"C":{"coverage":3,"factual_accuracy":3},"D":{"coverage":4,"factual_accuracy":4}}}')

keymap = json.load(open("results/gtr_packets/_keymap.json"))
CONDS = ["vec2summ_gmm", "medoid", "centroid", "random"]
SEEDS = [42, 43, 44]
JUDGES = {"codex": CODEX, "sonnet": SONNET}


def unblind(scores):
    """scores[cell][letter][crit] -> out[n][cond][crit] = list over seeds"""
    out = {1000: {c: {"coverage": [], "factual_accuracy": []} for c in CONDS},
           10000: {c: {"coverage": [], "factual_accuracy": []} for c in CONDS}}
    for cell, byletter in scores.items():
        _, n, _ = cell.split("_"); n = int(n)
        for letter, cond in keymap[cell].items():
            for crit in ("coverage", "factual_accuracy"):
                out[n][cond][crit].append(byletter[letter][crit])
    return out


print("GTR pipeline, trump, 3 seeds; coverage / factual_accuracy (mean over seeds)\n")
res = {}
for jname, sc in JUDGES.items():
    u = unblind(sc)
    print(f"=== judge: {jname} ===")
    print(f"{'cond':14} {'cov@1k':>7} {'cov@10k':>8} {'fact@1k':>8} {'fact@10k':>9}")
    for c in CONDS:
        cov1, cov10 = np.mean(u[1000][c]["coverage"]), np.mean(u[10000][c]["coverage"])
        fa1, fa10 = np.mean(u[1000][c]["factual_accuracy"]), np.mean(u[10000][c]["factual_accuracy"])
        print(f"{c:14} {cov1:>7.2f} {cov10:>8.2f} {fa1:>8.2f} {fa10:>9.2f}")
    g1 = np.mean(u[1000]["medoid"]["coverage"]) - np.mean(u[1000]["vec2summ_gmm"]["coverage"])
    g10 = np.mean(u[10000]["medoid"]["coverage"]) - np.mean(u[10000]["vec2summ_gmm"]["coverage"])
    print(f"  medoid-minus-Vec2Summ coverage gap:  n=1k {g1:+.2f}   n=10k {g10:+.2f}\n")
    res[jname] = {"gap_cov_1k": round(float(g1), 3), "gap_cov_10k": round(float(g10), 3),
                  "vec2summ_cov_1k": round(float(np.mean(u[1000]["vec2summ_gmm"]["coverage"])), 3),
                  "vec2summ_cov_10k": round(float(np.mean(u[10000]["vec2summ_gmm"]["coverage"])), 3),
                  "medoid_cov_1k": round(float(np.mean(u[1000]["medoid"]["coverage"])), 3),
                  "medoid_cov_10k": round(float(np.mean(u[10000]["medoid"]["coverage"])), 3)}

json.dump({"dataset": "trump", "judges": list(JUDGES), "result": res},
          open("results/tables/stats_gtr_quality.json", "w"), indent=1)
print("wrote results/tables/stats_gtr_quality.json")
