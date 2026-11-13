"""Medoid-k baseline + per-mode leakage audit (offline; cached embeddings).

medoid-k: the real corpus document nearest each GMM mode. Isolates "centrality"
(real doc near the mode) from "inversion" (Vec2Summ's snippet for that mode).
If medoid-k matches/beats Vec2Summ on probe coverage, inversion adds nothing.

leakage: per mode, effective support n*pi_j, cosine(mean_j, nearest real emb),
and text overlap between the inverted snippet_j (from e9) and its nearest doc.
High support-1 + high overlap = a mode memorizing an individual document.
"""
import os, json, re
# Deliberately NO OpenAI key: this analysis is cache-only. A cache miss must
# raise, never silently call the paid API.
os.environ["OPENAI_API_KEY"] = "BLOCKED-cache-only"
import numpy as np
from sklearn.mixture import GaussianMixture
import common as C

DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
K, N = 10, 1000
RES = C.EXP_DIR / "results"

def toks(s):
    return set(re.findall(r"\w+", s.lower()))

def jaccard(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0

def lcs_ratio(a, b):
    # char-level longest common substring / len(shorter), cheap DP on capped strings
    a, b = a[:600].lower(), b[:600].lower()
    if not a or not b:
        return 0.0
    prev = [0]*(len(b)+1); best = 0
    for i in range(1, len(a)+1):
        cur = [0]*(len(b)+1)
        ai = a[i-1]
        for j in range(1, len(b)+1):
            if ai == b[j-1]:
                cur[j] = prev[j-1]+1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best / min(len(a), len(b))

medoid_recon, support, near, overlap = {}, [], [], []
medoid_emb_cache = {}
for ds in DATASETS:
    mr = []
    for s in SEEDS:
        corpus = C.load_corpus(ds, N, s)
        emb = np.asarray(C.embed_openai(corpus, cache_tag=f"{ds}_{N}_{s}"), dtype=np.float32)
        gmm = GaussianMixture(K, covariance_type="diag", random_state=s).fit(emb)
        means = gmm.means_.astype(np.float32)
        w = gmm.weights_
        # cosine(means, emb)
        en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        mn = means / np.linalg.norm(means, axis=1, keepdims=True)
        sims = mn @ en.T                       # [K, n]
        medoid_idx = sims.argmax(axis=1)       # nearest real doc per mode
        medoid_emb = emb[medoid_idx]
        rq = C.recon_quality(emb, medoid_emb)  # mean-max cosine of medoid set
        mr.append(rq["mean_max_sim"])
        medoid_emb_cache[(ds, s)] = (medoid_idx, [corpus[i] for i in medoid_idx])
        # leakage: per-mode support + nearest-real cosine + snippet/doc overlap
        e9 = json.load(open(RES / "e9" / f"{ds}_{N}_{s}_gmm.json"))
        snippets = e9["conditions"]["vec2summ_gmm"]["texts"]
        for j in range(K):
            support.append(float(w[j]) * N)
            near.append(float(sims[j, medoid_idx[j]]))
            doc = corpus[medoid_idx[j]]
            overlap.append({"dataset": ds, "seed": s, "mode": j,
                            "support": float(w[j])*N,
                            "near_cos": float(sims[j, medoid_idx[j]]),
                            "jaccard": jaccard(snippets[j], doc),
                            "lcs_ratio": lcs_ratio(snippets[j], doc)})
    medoid_recon[ds] = float(np.mean(mr))

# load comparison recon (vec2summ_gmm, centroid, random) from artifacts
def recon_means():
    import csv
    rows = list(csv.DictReader(open(RES / "tables" / "recon_quality.csv")))
    cr = {r["dataset"]: float(r["centroid"]) for r in rows if r["n"] == "1000"}
    rr = {r["dataset"]: float(r["random"]) for r in rows if r["n"] == "1000"}
    gmm = {}
    for ds in DATASETS:
        v = [json.load(open(RES/"e9"/f"{ds}_{N}_{s}_gmm.json"))["conditions"]["vec2summ_gmm"]["recon"]["mean_max_sim"] for s in SEEDS]
        gmm[ds] = float(np.mean(v))
    return cr, rr, gmm
cr, rr, gmm = recon_means()
name = {"tifu": "tifu", "onebook": "onebook", "askdocs": "askdocs",
        "askdocs_fever": "askdocs_fever", "biden": "biden", "illegal": "illegal", "trump": "trump"}

out = {"medoid_recon": medoid_recon, "vec2summ_gmm_recon": gmm,
       "centroid_recon": {d: cr[d] for d in DATASETS},
       "random_recon": {d: rr[d] for d in DATASETS},
       "leakage": {
           "support_min": float(np.min(support)), "support_p05": float(np.percentile(support, 5)),
           "support_median": float(np.median(support)),
           "near_cos_max": float(np.max(near)), "near_cos_mean": float(np.mean(near)),
           "near_cos_p95": float(np.percentile(near, 95)),
           "jaccard_max": float(max(o["jaccard"] for o in overlap)),
           "jaccard_mean": float(np.mean([o["jaccard"] for o in overlap])),
           "lcs_max": float(max(o["lcs_ratio"] for o in overlap)),
           "lcs_mean": float(np.mean([o["lcs_ratio"] for o in overlap])),
           "n_modes": len(overlap),
       },
       "leakage_per_mode": overlap}
json.dump(out, open(RES/"tables"/"stats_medoid_leakage.json", "w"), indent=1)

# medoid vs gmm: per-dataset and macro
print("=== Probe coverage (mean-max cosine): medoid-k vs Vec2Summ vs baselines ===")
print(f"{'dataset':14s} {'medoid':>7s} {'v2s_gmm':>8s} {'centroid':>9s} {'random':>7s}")
dm = []
for ds in DATASETS:
    dm.append(medoid_recon[ds]-gmm[ds])
    print(f"{ds:14s} {medoid_recon[ds]:7.3f} {gmm[ds]:8.3f} {cr[ds]:9.3f} {rr[ds]:7.3f}")
print(f"{'MACRO':14s} {np.mean(list(medoid_recon.values())):7.3f} "
      f"{np.mean(list(gmm.values())):8.3f} {np.mean(list(cr.values())):9.3f} "
      f"{np.mean(list(rr.values())):7.3f}")
print(f"medoid - v2s_gmm: mean {np.mean(dm):+.4f}, wins(medoid>gmm) {sum(d>0 for d in dm)}/7")
print("\n=== Leakage audit (per-mode, 210 modes = 7x3x10) ===")
L = out["leakage"]
print(f"effective support n*pi_j: min {L['support_min']:.1f}, p05 {L['support_p05']:.1f}, median {L['support_median']:.1f}")
print(f"cosine(mode, nearest real doc): mean {L['near_cos_mean']:.3f}, p95 {L['near_cos_p95']:.3f}, max {L['near_cos_max']:.3f}")
print(f"snippet<->nearest-doc token Jaccard: mean {L['jaccard_mean']:.3f}, max {L['jaccard_max']:.3f}")
print(f"snippet<->nearest-doc char-LCS ratio: mean {L['lcs_mean']:.3f}, max {L['lcs_max']:.3f}")
# worst (most leaky) modes
worst = sorted(overlap, key=lambda o: o["jaccard"], reverse=True)[:5]
print("top-5 highest snippet/doc token overlap (potential regurgitation):")
for o in worst:
    print(f"  {o['dataset']}/{o['seed']}/m{o['mode']}: support {o['support']:.0f}, "
          f"near_cos {o['near_cos']:.2f}, jaccard {o['jaccard']:.2f}, lcs {o['lcs_ratio']:.2f}")
# save medoid texts for the G-Eval step
json.dump({f"{d}_{s}": medoid_emb_cache[(d, s)][1] for d in DATASETS for s in SEEDS},
          open(RES/"tables"/"medoid_texts.json", "w"))
print("\nwrote stats_medoid_leakage.json + medoid_texts.json")
