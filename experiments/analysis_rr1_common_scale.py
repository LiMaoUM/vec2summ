"""RR1 confound fix: medoid-k vs conditions on a SINGLE common scale
(Llama-3.3-70B summarizer, all 5 conditions A-E), judged by TWO independent
families: codex/GPT-5 (results/codex_judge) and Claude-Sonnet (results/medoid_sonnet).
Un-blind via codex_packets/_keymap.json. Reports per-condition means per judge +
paired sign-flip tests for medoid vs each baseline, per judge, plus cross-judge
agreement on the medoid-vs-Vec2Summ direction.
"""
import json, glob, random
from collections import defaultdict
from statistics import mean, stdev

RES = "results"
CRITS = ["coverage", "factual_accuracy", "coherence", "conciseness"]
CONDS = ["medoid", "vec2summ_gmm", "centroid", "random", "direct"]
rng = random.Random(23)
keymap = json.load(open(f"{RES}/codex_packets/_keymap.json"))

def load(judge_dir):
    sc = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(f"{RES}/{judge_dir}/*.json"):
        cell = f.split("/")[-1][:-5]
        if cell not in keymap:
            continue
        d = json.load(open(f)); inv = keymap[cell]
        for lbl, s in d.items():
            cond = inv[lbl]
            for c in CRITS:
                sc[cond][c].append(s[c]); paired[cell].setdefault(cond, {})[c] = s[c]
    return sc, paired

def signflip(diffs, n=20000):
    obs = mean(diffs)
    if all(d == 0 for d in diffs): return 1.0
    return sum(1 for _ in range(n) if abs(mean(d if rng.random()<.5 else -d for d in diffs)) >= abs(obs)-1e-12)/n

judges = {"codex": "codex_judge", "sonnet": "medoid_sonnet"}
out = {}
for jname, jdir in judges.items():
    sc, paired = load(jdir)
    means = {c: {cr: (mean(sc[c][cr]), stdev(sc[c][cr])) for cr in CRITS} for c in CONDS}
    cells = sorted(paired)
    tests = {}
    for base in ["vec2summ_gmm", "centroid", "random"]:
        for cr in ["coverage", "factual_accuracy"]:
            diffs = [paired[cell]["medoid"][cr]-paired[cell][base][cr] for cell in cells]
            tests[f"medoid_vs_{base}|{cr}"] = {"mean_diff": mean(diffs), "p": signflip(diffs)}
    out[jname] = {"means": {c: {cr: {"mean": m[0], "sd": m[1]} for cr, m in d.items()} for c, d in means.items()},
                  "tests": tests, "n": len(cells)}
json.dump(out, open(f"{RES}/tables/stats_rr1.json", "w"), indent=1)

print("=== medoid-k confound fix: ONE scale (Llama summaries), TWO judge families ===\n")
for jname in judges:
    m = out[jname]["means"]
    print(f"--- {jname} judge (n={out[jname]['n']} cells) ---")
    print(f"{'cond':14s} " + " ".join(f"{c[:6]:>7s}" for c in CRITS))
    for c in CONDS:
        print(f"{c:14s} " + " ".join(f"{m[c][cr]['mean']:.2f}" for cr in CRITS))
    for k, t in out[jname]["tests"].items():
        print(f"   {k:34s} diff {t['mean_diff']:+.2f} p={t['p']:.3f}")
    print()
print("=== cross-judge agreement on medoid vs Vec2Summ-gmm direction ===")
for cr in ["coverage", "factual_accuracy"]:
    cd = out["codex"]["tests"][f"medoid_vs_vec2summ_gmm|{cr}"]
    sd = out["sonnet"]["tests"][f"medoid_vs_vec2summ_gmm|{cr}"]
    agree = (cd["mean_diff"] > 0) == (sd["mean_diff"] > 0)
    print(f"{cr}: codex {cd['mean_diff']:+.2f}(p={cd['p']:.3f}), sonnet {sd['mean_diff']:+.2f}(p={sd['p']:.3f}) -> {'AGREE' if agree else 'DISAGREE'}")
