"""Can the sketch be built WITHOUT ever holding the whole corpus? (offline)
Two constructions that selection baselines cannot match:
 (1) streaming: single-pass online EM over 500-doc minibatches, O(kd) state.
 (2) shard-merge: fit a GMM per 1k-doc shard independently, merge the S*k
     component means into k by weighted k-means. Peak memory = one shard.
Compare probe coverage of each to the batch GMM that holds all n embeddings.
Uses the cached 10k GTR embeddings from scaling_curve.py. Writes stats_stream.json.
"""
import json, numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

DATASETS = ["askdocs", "trump", "tifu"]
SEEDS = [42, 43, 44]
N, K = 10000, 10
CACHE = Path("cache")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def covg(eu, modes):
    return float((eu @ unit(modes).T).max(axis=1).mean())


def batch_gmm(emb, seed):
    return GaussianMixture(K, covariance_type="diag", random_state=seed).fit(emb).means_


def online_em(emb, seed, chunk=500):
    """Single-pass stepwise online EM (Cappe-Moulines) for a diagonal GMM.
    State is O(K*d): means, vars, weights, never the full corpus."""
    rng = np.random.default_rng(seed)
    g0 = GaussianMixture(K, covariance_type="diag", random_state=seed).fit(emb[:1000])
    mu, var, w = g0.means_.copy(), g0.covariances_.copy(), g0.weights_.copy()
    d = emb.shape[1]
    S0, S1, S2 = w.copy(), mu * w[:, None], (var + mu ** 2) * w[:, None]
    order = rng.permutation(len(emb))            # stream order
    t = 0
    for start in range(0, len(emb), chunk):
        xb = emb[order[start:start + chunk]]
        # E-step: responsibilities under current diag-Gaussian params
        logdet = np.log(var).sum(1)
        diff2 = (xb[:, None, :] - mu[None]) ** 2 / var[None]
        logp = -0.5 * (diff2.sum(2) + logdet[None] + d * np.log(2 * np.pi)) + np.log(w + 1e-12)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp); r /= r.sum(1, keepdims=True)
        # minibatch per-point averages of sufficient stats
        n0 = r.mean(0); n1 = (r.T @ xb) / len(xb); n2 = (r.T @ (xb ** 2)) / len(xb)
        t += 1; gamma = (t + 2.0) ** -0.6        # decreasing step size
        S0 = (1 - gamma) * S0 + gamma * n0
        S1 = (1 - gamma) * S1 + gamma * n1
        S2 = (1 - gamma) * S2 + gamma * n2
        w = S0 / S0.sum()
        mu = S1 / S0[:, None]
        var = np.clip(S2 / S0[:, None] - mu ** 2, 1e-6, None)
    return mu


def shard_merge(emb, seed, shards=10):
    """Fit a GMM per shard independently, then merge S*K means into K by
    weight-weighted k-means. Peak memory = one shard (n/shards)."""
    idx = np.array_split(np.arange(len(emb)), shards)
    means, wts = [], []
    for j, ix in enumerate(idx):
        g = GaussianMixture(K, covariance_type="diag", random_state=seed + j).fit(emb[ix])
        means.append(g.means_); wts.append(g.weights_ * len(ix))
    means = np.vstack(means); wts = np.concatenate(wts)
    km = KMeans(K, random_state=seed, n_init=10).fit(means, sample_weight=wts)
    return km.cluster_centers_


rows = {}
for ds in DATASETS:
    for s in SEEDS:
        emb = np.load(CACHE / f"gtr_{ds}_{s}_{N}.npy").astype(np.float32)
        eu = unit(emb)
        rows.setdefault(ds, {"batch": [], "stream": [], "merge": []})
        rows[ds]["batch"].append(covg(eu, batch_gmm(emb, s)))
        rows[ds]["stream"].append(covg(eu, online_em(emb, s)))
        rows[ds]["merge"].append(covg(eu, shard_merge(emb, s)))

d = emb.shape[1]
agg = {ds: {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
            for k, v in r.items()} for ds, r in rows.items()}
mem = {"batch_floats": N * d, "stream_floats": K * d * 3 + 500 * d,
       "merge_peak_floats": (N // 10) * d}
out = {"datasets": DATASETS, "seeds": SEEDS, "n": N, "k": K, "d": d,
       "metric": "probe coverage (mean-max cosine), GTR", "per_dataset": agg, "memory": mem}
json.dump(out, open("results/tables/stats_stream.json", "w"), indent=1)

print(f"n={N} k={K} d={d}  (probe coverage; mean+/-sd over 3 seeds)\n")
print(f"{'dataset':10} {'batch':>14} {'stream(1-pass)':>16} {'shard-merge':>14}")
for ds in DATASETS:
    a = agg[ds]
    g = lambda k: f"{a[k][0]:.3f}±{a[k][1]:.3f}"
    print(f"{ds:10} {g('batch'):>14} {g('stream'):>16} {g('merge'):>14}")
print(f"\nmemory @ n={N}, d={d}: batch holds {mem['batch_floats']:,} floats; "
      f"streaming holds {mem['stream_floats']:,}; shard-merge peak {mem['merge_peak_floats']:,}")
print(f"streaming is {mem['batch_floats']/mem['stream_floats']:.0f}x smaller state than batch")
