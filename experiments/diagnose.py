"""Decompose the vec2summ quality gap and score candidate fixes, offline.

Stage attribution per cell:
  A. sampled-vector quality  : recon_quality(corpus, Gaussian samples) — before inversion
  B. inversion fidelity      : cosine(embed(inverted_text_i), sampled_vec_i)
  C. end quality             : recon_quality(corpus, embed(inverted texts)) — from e1

Candidate selection schemes scored as raw vectors (inversion-free upper bounds):
  mean_only      : the mean vector, k times (T=0 limit)
  gaussian_T1.2  : paper setting (= stage A)
  pca_spread     : mu +/- s*v_i along top components (structured, on-cone)
  gmm_means      : k-component GMM component means (diag cov)
  gmm_medoids    : nearest real embedding to each GMM mean
  kmeans_medoids : nearest real embedding to each k-means center
Medoids are real points, so their inversion is near-lossless; their vector
score is a tight proxy for the full-pipeline score.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

K = 10
CELLS = [("trump", 1000, 42), ("onebook", 1000, 42), ("askdocs", 1000, 42)]


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def medoids(emb, centers):
    sims = norm(centers) @ norm(emb).T
    return emb[np.unique(sims.argmax(axis=1))]


def main():
    for ds, n, seed in CELLS:
        corpus = C.load_corpus(ds, n, seed)
        emb = C.embed_openai(corpus, cache_tag=f"{ds}_{n}_{seed}")
        mu, cov = C.fit_gaussian(emb)
        gauss = C.sample_gaussian(mu, cov, K, 1.2, seed)

        # candidate schemes (vector-level)
        schemes = {"mean_only": mu[None, :], "gaussian_T1.2": gauss}
        evals, evecs = np.linalg.eigh(cov)
        top = evecs[:, -5:]                       # top-5 principal directions
        sds = np.sqrt(evals[-5:])
        schemes["pca_spread"] = np.vstack([mu + s * v for s, v in zip(sds, top.T)]
                                          + [mu - s * v for s, v in zip(sds, top.T)])
        gmm = GaussianMixture(K, covariance_type="diag", random_state=seed).fit(emb)
        schemes["gmm_means"] = gmm.means_
        schemes["gmm_medoids"] = medoids(emb, gmm.means_)
        km = KMeans(K, random_state=seed, n_init=4).fit(emb)
        schemes["kmeans_medoids"] = medoids(emb, km.cluster_centers_)

        print(f"\n=== {ds} n={n} seed={seed}")
        for name, vecs in schemes.items():
            q = C.recon_quality(emb, np.asarray(vecs, dtype=np.float32))
            print(f"  {name:15s} mean_max_sim={q['mean_max_sim']:.4f}  (k={len(vecs)})")

        # stage B: inversion fidelity from the existing artifact
        art = json.loads((C.EXP_DIR / "results/e1" / f"{ds}_{n}_{seed}.json").read_text())
        texts = art["conditions"]["vec2summ"]["texts"]
        remb = C.embed_openai(texts)
        fid = (norm(remb) * norm(gauss[: len(remb)])).sum(axis=1)
        end = art["conditions"]["vec2summ"]["recon"]["mean_max_sim"]
        print(f"  inversion fidelity cos(text_i, vec_i): mean={fid.mean():.4f} min={fid.min():.4f}")
        print(f"  end-to-end (from e1): {end:.4f}")


if __name__ == "__main__":
    main()
