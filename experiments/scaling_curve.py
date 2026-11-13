"""Scaling curve (offline, GTR encoder, no API key): does the O(kd) sketch keep
covering the corpus as n grows past the context window?
3 datasets x 3 seeds x n in {1k,2k,5k,10k}. Geometry only: mean-max cosine of
corpus docs to k probe/selection vectors. Also the Direct full-context token cost.
For each (dataset,seed) we embed 10k once (cached) and use nested prefixes for
smaller n. Writes results/tables/stats_scaling.json.
"""
import os, json, numpy as np
os.environ["OPENAI_API_KEY"] = "BLOCKED-cache-only"
from pathlib import Path
from sklearn.mixture import GaussianMixture
import common as C
from e3_gtr import embed_gtr

DATASETS = ["askdocs", "trump", "tifu"]
SEEDS = [42, 43, 44]
NS = [1000, 2000, 5000, 10000]
NMAX, K = 10000, 10
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def covg(corpus_u, cand_u):
    return float((corpus_u @ cand_u.T).max(axis=1).mean())


def embed_cached(docs, tag):
    f = CACHE / f"gtr_{tag}_{len(docs)}.npy"
    if f.exists():
        return np.load(f)
    e = np.asarray(embed_gtr(docs), dtype=np.float32)
    np.save(f, e)
    return e


cells = {}  # (dataset, n) -> {cond: [seed values]}
for ds in DATASETS:
    for s in SEEDS:
        docs10k = C.load_corpus(ds, NMAX, s)
        emb10k = embed_cached(docs10k, f"{ds}_{s}")
        for n in NS:
            docs, emb = docs10k[:n], emb10k[:n]
            eu = unit(emb)
            gmm = GaussianMixture(K, covariance_type="diag", random_state=s).fit(emb)
            modes_u = unit(gmm.means_.astype(np.float32))
            rng = np.random.default_rng(s)
            rand_u = eu[rng.choice(n, K, replace=False)]
            cmean = unit(emb.mean(0, keepdims=True))
            cent_u = eu[np.argsort(-(eu @ cmean.T).ravel())[:K]]
            med_idx = (modes_u @ eu.T).argmax(axis=1)
            row = {"gmm": covg(eu, modes_u), "random": covg(eu, rand_u),
                   "centroid": covg(eu, cent_u), "medoid": covg(eu, eu[med_idx]),
                   "direct_tok": int(sum(len(d) for d in docs) / 4),
                   "v2s_tok": int(sum(len(docs[i]) for i in med_idx) / 4)}
            cells.setdefault((ds, n), {}).setdefault("_rows", []).append(row)

# aggregate mean +/- sd over seeds
agg = {}
for (ds, n), d in cells.items():
    rows = d["_rows"]
    agg.setdefault(ds, {})[n] = {
        k: [round(float(np.mean([r[k] for r in rows])), 4),
            round(float(np.std([r[k] for r in rows])), 4)]
        for k in rows[0]}

out = {"datasets": DATASETS, "seeds": SEEDS, "ns": NS, "k": K,
       "encoder": "gtr-t5-base", "metric": "mean-max cosine (probe-vector coverage)",
       "per_dataset": agg}
json.dump(out, open("results/tables/stats_scaling.json", "w"), indent=1)

print(f"{'dataset':10} {'n':>6} {'gmm':>13} {'random':>13} {'centroid':>13} "
      f"{'medoid':>13} {'direct_tok':>11} {'v2s_tok':>8}")
for ds in DATASETS:
    for n in NS:
        a = agg[ds][n]
        f = lambda k: f"{a[k][0]:.3f}±{a[k][1]:.3f}"
        print(f"{ds:10} {n:>6} {f('gmm'):>13} {f('random'):>13} {f('centroid'):>13} "
              f"{f('medoid'):>13} {a['direct_tok'][0]:>11,.0f} {a['v2s_tok'][0]:>8,.0f}")
