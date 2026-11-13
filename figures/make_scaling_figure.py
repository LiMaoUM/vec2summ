"""Figure: scaling to n=10k (Appendix E).
(a) probe coverage vs n: the k=10 sketch covers a 10x larger corpus as well as a
    small one, and tracks the document-selection baselines at every scale.
(b) cost vs n: Direct full-context input grows linearly past the context window,
    while the sketch's summarizer cost stays flat.
Reads revision/experiments/results/tables/stats_scaling.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent.parent / "experiments"
d = json.load(open(EXP / "results/tables/stats_scaling.json"))
DS, NS = d["datasets"], d["ns"]
INK = "#22223b"
COL = {"gmm": "#3f6fb0", "medoid": "#4f8a6b", "centroid": "#c79a3a", "random": "#cf4d5e"}
LAB = {"gmm": "Vec2Summ modes", "medoid": "medoid-$k$", "centroid": "centroid-$k$", "random": "random-$k$"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})


def mean_over_ds(cond, stat=0):
    # mean across datasets of the per-dataset seed-mean, per n
    return np.array([np.mean([d["per_dataset"][ds][str(n)][cond][stat] for ds in DS]) for n in NS])


def sd_over_ds(cond):
    return np.array([np.std([d["per_dataset"][ds][str(n)][cond][0] for ds in DS]) for n in NS])


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# ---- (a) coverage vs n ----
for cond in ["gmm", "medoid", "centroid", "random"]:
    m, s = mean_over_ds(cond), sd_over_ds(cond)
    axA.plot(NS, m, "-o", color=COL[cond], lw=1.6, ms=4, label=LAB[cond], zorder=3)
    axA.fill_between(NS, m - s, m + s, color=COL[cond], alpha=0.12, lw=0, zorder=1)
axA.set_xscale("log")
axA.set_xticks(NS); axA.set_xticklabels([f"{n//1000}k" for n in NS])
axA.set_xlabel("corpus size $n$", color=INK)
axA.set_ylabel("probe coverage (mean-max cosine)", color=INK)
axA.set_title("(a) coverage is stable with scale", fontsize=9.5, color=INK)
axA.grid(True, color="#e1e6ee", lw=0.7, zorder=0); axA.set_axisbelow(True)
axA.legend(fontsize=7.6, loc="center right", frameon=True, framealpha=0.9,
           handletextpad=0.4, borderpad=0.4)

# ---- (b) cost vs n ----
direct = mean_over_ds("direct_tok")
v2s = mean_over_ds("v2s_tok")
axB.plot(NS, direct, "-o", color="#cf4d5e", lw=1.8, ms=4, label="Direct (full context)", zorder=4)
axB.plot(NS, v2s, "-o", color="#3f6fb0", lw=1.8, ms=4, label="Vec2Summ (sketch)", zorder=4)
for y, t in [(128000, "128k ctx"), (1000000, "1M ctx")]:
    axB.axhline(y, color="#9aa3b2", lw=0.9, ls=(0, (4, 3)), zorder=1)
    axB.text(NS[0], y * 1.15, t, fontsize=7.2, color="#6b7280")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xticks(NS); axB.set_xticklabels([f"{n//1000}k" for n in NS])
axB.set_xlabel("corpus size $n$", color=INK)
axB.set_ylabel("summarizer input tokens", color=INK)
axB.set_title("(b) full-context cost outgrows the window", fontsize=9.5, color=INK)
axB.grid(True, which="both", color="#e1e6ee", lw=0.6, zorder=0); axB.set_axisbelow(True)
axB.legend(fontsize=7.8, loc="center right", frameon=True, framealpha=0.9,
           handletextpad=0.4, borderpad=0.4)

fig.tight_layout(pad=0.6)
out = Path(__file__).parent / "figures" / "scaling.png"
fig.savefig(out, dpi=500, bbox_inches="tight", facecolor="white")
print("wrote", out)
print("coverage gmm @1k,10k:", round(mean_over_ds("gmm")[0], 3), round(mean_over_ds("gmm")[-1], 3))
print("direct tok @1k,10k:", int(direct[0]), int(direct[-1]), "| v2s @10k:", int(v2s[-1]))
