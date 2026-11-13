"""Corrected sketch-leakage threshold analysis (supersedes the v1 sweep whose
reported residual/threshold pair did not reproduce). Everything below is computed
at the EXACT operating point used (near_cos tau = 0.959), and the coverage cost is
refit per cell from cached embeddings. Writes results/tables/stats_threshold.json.

Run from revision/experiments/ with OPENAI_API_KEY=BLOCKED-cache-only (cache only).
"""
import os, json
import numpy as np
from statistics import mean
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import common as C

DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
TWITTER = {"biden", "trump", "illegal"}
SEEDS = [42, 43, 44]
TAU = 0.959  # suppress modes whose nearest real-document cosine exceeds TAU

pm = json.load(open("results/tables/stats_medoid_leakage.json"))["leakage_per_mode"]
n = len(pm)


def corr(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


supp = [m["support"] for m in pm]
nc = [m["near_cos"] for m in pm]
jac = [m["jaccard"] for m in pm]

# correlations, overall and split by domain (leakage is a redundant-Twitter effect)
tw = [m for m in pm if m["dataset"] in TWITTER]
ntw = [m for m in pm if m["dataset"] not in TWITTER]

# what the near_cos threshold does, at the exact tau (not on a 2-decimal grid)
kept = [m for m in pm if m["near_cos"] <= TAU]
cut = n - len(kept)
resid_max_jac = max(m["jaccard"] for m in kept)
hi4_left = sum(1 for m in kept if m["jaccard"] >= 0.4)

# support threshold needed to clear the same Jaccard>=0.4 tail
sup_to_clear = max(m["support"] for m in pm if m["jaccard"] >= 0.4)
sup_cut = sum(1 for m in pm if m["support"] < sup_to_clear)

# probe-coverage cost, refit per cell (mean-max cosine, all modes vs retained modes)
def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

per_ds_cov, per_ds_cut = {}, {}
full_all, keep_all = [], []
for ds in DATASETS:
    fs, ks = [], []
    for s in SEEDS:
        emb = np.asarray(C.embed_openai(C.load_corpus(ds, 1000, s),
                                        cache_tag=f"{ds}_1000_{s}"), dtype=np.float32)
        gmm = GaussianMixture(10, covariance_type="diag", random_state=s).fit(emb)
        en, mn = unit(emb), unit(gmm.means_.astype(np.float32))
        near = (mn @ en.T).max(1)
        keep_mask = near <= TAU
        if keep_mask.sum() == 0:
            keep_mask[near.argmin()] = True
        fs.append(float((en @ mn.T).max(1).mean()))
        ks.append(float((en @ mn[keep_mask].T).max(1).mean()))
    per_ds_cov[ds] = [round(mean(fs), 4), round(mean(ks), 4)]
    full_all += fs
    keep_all += ks
    ms = [m for m in pm if m["dataset"] == ds]
    per_ds_cut[ds] = round(100 * sum(1 for m in ms if m["near_cos"] > TAU) / len(ms))

out = {
    "n_modes": n,
    "tau_near_cos": TAU,
    "corr_support_jaccard": round(corr(supp, jac), 3),
    "corr_nearcos_jaccard": round(corr(nc, jac), 3),
    "corr_support_nearcos": round(corr(supp, nc), 3),
    "corr_nearcos_jaccard_twitter": round(corr([m["near_cos"] for m in tw], [m["jaccard"] for m in tw]), 3),
    "corr_nearcos_jaccard_nontwitter": round(corr([m["near_cos"] for m in ntw], [m["jaccard"] for m in ntw]), 3),
    "nontwitter_max_jaccard": round(max(m["jaccard"] for m in ntw), 3),
    "high_leak_modes_jac_ge_0.4": [
        {"cell": f"{m['dataset']}/{m['seed']}/m{m['mode']}", "jac": round(m["jaccard"], 3),
         "support": round(m["support"]), "near_cos": round(m["near_cos"], 3)}
        for m in sorted([m for m in pm if m["jaccard"] >= 0.4], key=lambda x: -x["jaccard"])],
    "nearcos_threshold": {
        "tau": TAU,
        "modes_cut_pct": round(100 * cut / n, 1),
        "resid_max_jaccard": round(resid_max_jac, 3),
        "jac_ge_0.4_surviving": hi4_left,
        "probe_cov_all": round(mean(full_all), 3),
        "probe_cov_after": round(mean(keep_all), 3),
        "coverage_drop_pct": round(100 * (mean(full_all) - mean(keep_all)) / mean(full_all), 1),
        "in_sample_tuned": True,
    },
    "support_threshold_to_clear_jac0.4": {
        "tau": round(sup_to_clear), "modes_cut_pct": round(100 * sup_cut / n)},
    "per_dataset_coverage_all_kept": per_ds_cov,
    "per_dataset_pct_modes_cut": per_ds_cut,
}
json.dump(out, open("results/tables/stats_threshold.json", "w"), indent=1)
print(json.dumps(out, indent=1))
