"""Aggregate the codex (GPT-5-family) judge over the blinded Llama-summary set,
un-blinded via codex_packets/_keymap.json. Reports per-condition means and the
decisive medoid-k vs Vec2Summ-gmm comparison with a paired sign-flip test.
This is the summary-quality counterpart to the offline probe-coverage tie.
"""
import json, glob, math, random
from collections import defaultdict
from statistics import mean, stdev

RES = "results"
CRITS = ["coverage", "factual_accuracy", "coherence", "conciseness"]
CONDS = ["medoid", "vec2summ_gmm", "centroid", "random", "direct"]
rng = random.Random(11)
keymap = json.load(open(f"{RES}/codex_packets/_keymap.json"))

scores = defaultdict(lambda: defaultdict(list))   # cond -> crit -> [cell values]
paired = defaultdict(lambda: defaultdict(dict))   # cell -> cond -> crit (for paired tests)
for f in glob.glob(f"{RES}/codex_judge/*.json"):
    cell = f.split("/")[-1][:-5]
    d = json.load(open(f))
    inv = keymap[cell]
    for lbl, sc in d.items():
        cond = inv[lbl]
        for c in CRITS:
            scores[cond][c].append(sc[c])
            paired[cell].setdefault(cond, {})[c] = sc[c]

means = {cond: {c: (mean(scores[cond][c]), stdev(scores[cond][c])) for c in CRITS} for cond in CONDS}

def signflip_p(diffs, n=20000):
    obs = mean(diffs)
    if all(d == 0 for d in diffs):
        return 1.0
    cnt = sum(1 for _ in range(n)
              if abs(mean(d if rng.random() < .5 else -d for d in diffs)) >= abs(obs)-1e-12)
    return cnt/n

cells = sorted(paired)
tests = {}
for base in ["vec2summ_gmm", "centroid", "random"]:
    for c in ["coverage", "factual_accuracy"]:
        diffs = [paired[cell]["medoid"][c] - paired[cell][base][c] for cell in cells]
        tests[f"medoid_vs_{base}|{c}"] = {"mean_diff": mean(diffs), "p": signflip_p(diffs)}

out = {"codex_means": {cond: {c: {"mean": m[0], "sd": m[1]} for c, m in d.items()} for cond, d in means.items()},
       "medoid_tests": tests, "n_cells": len(cells)}
json.dump(out, open(f"{RES}/tables/stats_medoid_codex.json", "w"), indent=1)

print(f"=== Codex (GPT-5-family) judge, Llama summaries, {len(cells)} cells ===")
print(f"{'cond':14s} " + " ".join(f"{c[:6]:>11s}" for c in CRITS))
for cond in CONDS:
    print(f"{cond:14s} " + " ".join(f"{means[cond][c][0]:.2f}±{means[cond][c][1]:.2f}" for c in CRITS))
print("\n=== medoid-k vs baselines (paired sign-flip over cells) ===")
for k, t in tests.items():
    print(f"{k:34s} diff {t['mean_diff']:+.2f}  p={t['p']:.3f}")
