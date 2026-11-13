"""Coverage cost of the nearest-neighbour suppression safeguard (cache only).
For each cell: refit the GMM, flag modes whose mean sits within cosine TAU of a
real embedding (the leaky modes), and compare probe coverage (mean-max cosine of
corpus docs to mode means) using all k modes vs. the surviving modes.
"""
import os
os.environ["OPENAI_API_KEY"] = "BLOCKED-cache-only"
import numpy as np
from sklearn.mixture import GaussianMixture
import common as C

DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
TAU = 0.959  # suppress modes with nearest-real-doc cosine above this

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

full, kept_cov, removed_frac = [], [], []
for ds in DATASETS:
    for s in SEEDS:
        emb = np.asarray(C.embed_openai(C.load_corpus(ds, 1000, s),
                                        cache_tag=f"{ds}_1000_{s}"), dtype=np.float32)
        gmm = GaussianMixture(10, covariance_type="diag", random_state=s).fit(emb)
        means = gmm.means_.astype(np.float32)
        en, mn = unit(emb), unit(means)
        sims = mn @ en.T                      # [k, n] cosine mode->docs
        near = sims.max(axis=1)               # nearest-real-doc cosine per mode
        keep = near <= TAU
        # probe coverage = mean over docs of max cosine to any retained mode
        cov_all = (en @ mn.T).max(axis=1).mean()
        if keep.sum() == 0:
            keep[near.argmin()] = True        # never empty the set
        cov_keep = (en @ mn[keep].T).max(axis=1).mean()
        full.append(float(cov_all)); kept_cov.append(float(cov_keep))
        removed_frac.append(1 - keep.mean())

print(f"=== Nearest-neighbour suppression (near_cos > {TAU}) ===")
print(f"modes removed: macro {np.mean(removed_frac)*100:.0f}% "
      f"(range {min(removed_frac)*100:.0f}-{max(removed_frac)*100:.0f}%)")
print(f"probe coverage, all modes:      {np.mean(full):.3f}")
print(f"probe coverage, after suppress: {np.mean(kept_cov):.3f}")
print(f"coverage drop: {np.mean(full)-np.mean(kept_cov):+.4f} "
      f"({100*(np.mean(full)-np.mean(kept_cov))/np.mean(full):.1f}%)")
