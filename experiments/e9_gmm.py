"""E9: improved Vec2Summ variants — GMM-mode and PCA-spread probing.

Diagnosis (diagnose.py): the single-Gaussian *sampling* step is the dominant
quality loss — in 1536-d, samples from N(mu, 1.2*Sigma) live on a far-off
shell (vector-level 0.71-0.81), and the corpus is multi-modal so even mu is
suboptimal. Deterministic probes of the sketch score far higher as vectors:
GMM component means 0.88-0.94, PCA spread 0.87-0.94.

E9 runs the full pipeline on those schemes:
  gmm    : k-component diag-cov GMM on corpus embeddings -> invert the k means
  pca    : mu +/- sd_i * v_i along top-5 principal axes -> invert the 10 probes
Both remain document-free sketches (O(kd) / O(d^2) parameters).

Artifacts: results/e9/{ds}_{n}_{seed}_{scheme}.json . Resumable.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from e1_run import invert, local_up, summaries_for  # noqa: E402

RESULTS = C.EXP_DIR / "results"
K = 10


def scheme_vectors(scheme, emb, seed):
    mu, cov = C.fit_gaussian(emb)
    if scheme == "gmm":
        gmm = GaussianMixture(K, covariance_type="diag", random_state=seed).fit(emb)
        return gmm.means_
    if scheme == "pca":
        evals, evecs = np.linalg.eigh(cov)
        top, sds = evecs[:, -5:], np.sqrt(evals[-5:])
        return np.vstack([mu + s * v for s, v in zip(sds, top.T)]
                         + [mu - s * v for s, v in zip(sds, top.T)])
    raise ValueError(scheme)


def run_cell(ds, n, seed, scheme, use_local):
    out_f = RESULTS / "e9" / f"{ds}_{n}_{seed}_{scheme}.json"
    if out_f.exists():
        return "skip"
    corpus = C.load_corpus(ds, n, seed)
    emb = C.embed_openai(corpus, cache_tag=f"{ds}_{n}_{seed}")
    vecs = scheme_vectors(scheme, emb, seed)
    texts = invert(vecs)
    remb = C.embed_openai(texts)
    fid = float(np.mean(np.sum(
        (remb / np.linalg.norm(remb, axis=1, keepdims=True))
        * (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)), axis=1)))
    cell = {
        "dataset": ds, "n": n, "seed": seed, "k": len(vecs), "scheme": scheme,
        "vector_level": C.recon_quality(emb, np.asarray(vecs, dtype=np.float32)),
        "inversion_fidelity": fid,
        "conditions": {f"vec2summ_{scheme}": {
            "texts": texts,
            "recon": C.recon_quality(emb, remb),
            "summaries": summaries_for(texts, n, use_local),
        }},
    }
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(cell, indent=1))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    ap.add_argument("--sizes", nargs="*", type=int, default=[1000])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--schemes", nargs="*", default=["gmm", "pca"])
    args = ap.parse_args()
    use_local = local_up()
    print(f"[e9] local llama up: {use_local}", flush=True)
    for ds in args.datasets:
        for n in args.sizes:
            for seed in args.seeds:
                for sc in args.schemes:
                    try:
                        print(f"[e9] {ds} n={n} seed={seed} {sc}: "
                              f"{run_cell(ds, n, seed, sc, use_local)}", flush=True)
                    except Exception as e:
                        print(f"[e9] {ds} n={n} seed={seed} {sc}: ERROR {e!r}", flush=True)


if __name__ == "__main__":
    main()
