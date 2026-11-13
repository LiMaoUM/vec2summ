"""Does a support threshold actually suppress the sketch-leakage tail?
Honest audit over the 210 already-measured modes (stats_medoid_leakage.json):
which per-mode quantity predicts leakage, and which threshold removes the
high-overlap modes vs. only the tiny-support ones.
"""
import json
from statistics import mean

RES = "results" if __import__("os").path.isdir("results") else "."
pm = json.load(open(f"{RES}/tables/stats_medoid_leakage.json"))["leakage_per_mode"]
n = len(pm)

def corr(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0

supp = [m["support"] for m in pm]
ncos = [m["near_cos"] for m in pm]
jac = [m["jaccard"] for m in pm]
lcs = [m["lcs_ratio"] for m in pm]

print(f"=== {n} modes; what predicts snippet/doc token Jaccard? ===")
print(f"corr(support, jaccard)   = {corr(supp, jac):+.3f}")
print(f"corr(near_cos, jaccard)  = {corr(ncos, jac):+.3f}")
print(f"corr(support, near_cos)  = {corr(supp, ncos):+.3f}")
print(f"corr(near_cos, lcs)      = {corr(ncos, lcs):+.3f}")
print(f"corr(support, lcs)       = {corr(supp, lcs):+.3f}")

# define the "high-leakage" modes we actually care about removing
for thr in (0.4, 0.5):
    hi = [m for m in pm if m["jaccard"] >= thr]
    print(f"\nHigh-leak modes (Jaccard >= {thr}): {len(hi)} of {n}")
    for m in sorted(hi, key=lambda x: -x["jaccard"]):
        print(f"  {m['dataset']}/{m['seed']}/m{m['mode']}: jac {m['jaccard']:.2f}, "
              f"support {m['support']:.0f}, near_cos {m['near_cos']:.3f}")

def sweep(key, taus, drop_if_below=True, label=""):
    print(f"\n=== {label}: threshold on {key} ===")
    print(f"{'tau':>7} {'modes_cut':>10} {'%cut':>6} {'resid_maxJac':>13} "
          f"{'#Jac>=.4 left':>13} {'#Jac>=.3 left':>13}")
    for t in taus:
        if drop_if_below:
            kept = [m for m in pm if m[key] >= t]     # suppress modes BELOW tau (support)
        else:
            kept = [m for m in pm if m[key] <= t]     # suppress modes ABOVE tau (near_cos)
        cut = n - len(kept)
        mj = max((m["jaccard"] for m in kept), default=0.0)
        h4 = sum(1 for m in kept if m["jaccard"] >= 0.4)
        h3 = sum(1 for m in kept if m["jaccard"] >= 0.3)
        print(f"{t:>7.2f} {cut:>10} {100*cut/n:>5.0f}% {mj:>13.3f} {h4:>13} {h3:>13}")

# support threshold (the claim in the paper): suppress modes with support < tau
sweep("support", [0, 5, 10, 20, 36, 50, 75, 100], drop_if_below=True,
      label="SUPPORT threshold (paper's proposed safeguard)")
# nearest-neighbour threshold: suppress modes whose mean sits too close to a real doc
sweep("near_cos", [1.01, 0.99, 0.97, 0.96, 0.95, 0.93, 0.90], drop_if_below=False,
      label="NEAR-COS threshold (suppress modes too close to a real embedding)")

# how much does each safeguard cost in modes, to reach resid maxJac < 0.4?
print("\n=== minimal cut to push residual max-Jaccard below 0.4 ===")
for key, asc in (("support", True), ("near_cos", False)):
    best = None
    for m in sorted(pm, key=lambda x: x[key], reverse=not asc):
        pass
    # greedily: smallest cut that leaves max Jaccard < 0.4
    if asc:  # support: try increasing tau
        for t in sorted(set(round(m["support"]) for m in pm)):
            kept = [m for m in pm if m["support"] >= t]
            if kept and max(m["jaccard"] for m in kept) < 0.4:
                best = (t, n - len(kept)); break
    else:    # near_cos: try decreasing tau
        for t in sorted(set(round(m["near_cos"], 3) for m in pm), reverse=True):
            kept = [m for m in pm if m["near_cos"] <= t]
            if kept and max(m["jaccard"] for m in kept) < 0.4:
                best = (t, n - len(kept)); break
    if best:
        print(f"  {key}: tau={best[0]}, removes {best[1]}/{n} modes ({100*best[1]/n:.0f}%)")
    else:
        print(f"  {key}: no threshold removes all Jaccard>=0.4 modes without emptying the set")
