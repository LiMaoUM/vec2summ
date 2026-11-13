"""Figure: sketch-leakage diagnostic (Appendix C).
Two panels over the 210 audited modes (stats_medoid_leakage.json):
 (a) snippet-document Jaccard vs effective support  -> no relationship (r=-0.22)
 (b) Jaccard vs nearest-document cosine, split Twitter / non-Twitter,
     with the tau=0.959 suppression threshold -> the leak tracks proximity (+0.46).
Matches the Figure 1 palette/typography. No model calls; pure JSON.
"""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent.parent / "experiments"
pm = json.load(open(EXP / "results/tables/stats_medoid_leakage.json"))["leakage_per_mode"]
TWITTER = {"biden", "trump", "illegal"}
TAU = 0.959

INK = "#22223b"
BLUE = "#3f6fb0"
RED = "#cf4d5e"
GRID = "#d8dee8"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})


def corr(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


supp = [m["support"] for m in pm]
nc = [m["near_cos"] for m in pm]
jac = [m["jaccard"] for m in pm]
istw = [m["dataset"] in TWITTER for m in pm]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax in (axA, axB):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#c2cad6")
    ax.set_ylim(-0.03, 0.83)
    ax.set_ylabel("snippet--document Jaccard", color=INK)
    ax.axhline(0.4, color="#9aa3b2", lw=0.8, ls=(0, (4, 3)), zorder=1)


def scatter(ax, x):
    ax.scatter([x[i] for i in range(len(pm)) if not istw[i]],
               [jac[i] for i in range(len(pm)) if not istw[i]],
               s=20, facecolors="none", edgecolors=BLUE, lw=0.9, alpha=0.8,
               zorder=3, label="non-Twitter")
    ax.scatter([x[i] for i in range(len(pm)) if istw[i]],
               [jac[i] for i in range(len(pm)) if istw[i]],
               s=20, c=RED, lw=0, alpha=0.75, zorder=4, label="Twitter")


# ---- panel A: support ----
scatter(axA, supp)
axA.set_xlabel(r"effective support $n\hat\pi_j$ (documents)", color=INK)
axA.set_title("(a) leakage vs. support", fontsize=9.5, color=INK)
axA.text(0.96, 0.95, r"$r=-0.22$", transform=axA.transAxes, ha="right", va="top",
         fontsize=9.5, color=INK, fontweight="bold")
axA.text(0.96, 0.86, "no relationship", transform=axA.transAxes, ha="right",
         va="top", fontsize=8, color="#6b7280", style="italic")

# ---- panel B: nearest-document cosine ----
scatter(axB, nc)
axB.set_xlabel("nearest-document cosine", color=INK)
axB.set_title("(b) leakage vs. document proximity", fontsize=9.5, color=INK)
axB.axvline(TAU, color=INK, lw=1.1, ls=(0, (5, 2)), zorder=2)
axB.text(TAU - 0.004, 0.80, r"suppress $>\tau{=}0.959$", rotation=90, ha="right",
         va="top", fontsize=7.6, color=INK)
axB.text(0.04, 0.95, r"$r=+0.46$", transform=axB.transAxes, ha="left", va="top",
         fontsize=9.5, color=INK, fontweight="bold")
axB.text(0.04, 0.86, "within Twitter: $+0.66$", transform=axB.transAxes, ha="left",
         va="top", fontsize=8, color="#6b7280", style="italic")
axB.legend(loc="lower right", fontsize=8, frameon=True, handletextpad=0.3,
           borderpad=0.4, framealpha=0.9)

fig.tight_layout(pad=0.6)
out = Path(__file__).parent / "figures" / "leakage_diagnostic.png"
fig.savefig(out, dpi=500, bbox_inches="tight", facecolor="white")
print("wrote", out, "| r(support)=%.3f r(near_cos)=%.3f" % (corr(supp, jac), corr(nc, jac)))
