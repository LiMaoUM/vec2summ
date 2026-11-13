"""Proof-of-concept scaling probe (offline, GTR encoder, no API key).
Does the O(kd) sketch still cover the corpus as well as keeping k documents
when n grows past what fits in a context window? Geometry only (no inversion/LLM):
mean-max cosine of corpus docs to k probe/selection vectors, for n in {1k,5k,10k}.
Also reports the Direct full-context token cost that motivates the regime.
"""
import os, sys, numpy as np
os.environ["OPENAI_API_KEY"] = "BLOCKED-cache-only"
from sklearn.mixture import GaussianMixture
import common as C
from e3_gtr import embed_gtr

DATASET = sys.argv[1] if len(sys.argv) > 1 else "askdocs"
NS = [1000, 5000, 10000]
K, SEED = 10, 42


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def covg(corpus_u, cand_u):
    return float((corpus_u @ cand_u.T).max(axis=1).mean())


print(f"dataset={DATASET}  k={K}  seed={SEED}  encoder=gtr-t5-base\n")
print(f"{'n':>7} {'gmm':>7} {'random':>7} {'centroid':>9} {'medoid':>7} "
      f"{'direct_tok':>11} {'v2s_tok':>8}")
for n in NS:
    docs = C.load_corpus(DATASET, n, SEED)
    emb = np.asarray(embed_gtr(docs), dtype=np.float32)
    eu = unit(emb)
    gmm = GaussianMixture(K, covariance_type="diag", random_state=SEED).fit(emb)
    modes_u = unit(gmm.means_.astype(np.float32))
    rng = np.random.default_rng(SEED)
    rand_u = eu[rng.choice(n, K, replace=False)]
    cmean = unit(emb.mean(0, keepdims=True))
    cent_u = eu[np.argsort(-(eu @ cmean.T).ravel())[:K]]
    medoid_idx = (modes_u @ eu.T).argmax(axis=1)        # real doc nearest each mode
    med_u = eu[medoid_idx]
    # Direct full-context cost ~ total document tokens (chars/4); Vec2Summ ~ k snippets
    direct_tok = int(sum(len(d) for d in docs) / 4)
    v2s_tok = int(sum(len(docs[i]) for i in medoid_idx) / 4)  # k snippets, proxy
    print(f"{n:>7} {covg(eu, modes_u):>7.3f} {covg(eu, rand_u):>7.3f} "
          f"{covg(eu, cent_u):>9.3f} {covg(eu, med_u):>7.3f} "
          f"{direct_tok:>11,} {v2s_tok:>8,}")
