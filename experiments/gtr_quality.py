"""End-to-end judged-quality at scale, GTR pipeline (offline summaries).
For trump at n in {1k,10k}, 3 seeds: build the four cost-matched conditions and
summarize each with Llama-3.3-70B (vLLM). Same encoder at both n, so the only
thing varying is scale -> isolates whether the document-free gap widens at 10k.
Direct is omitted: at 10k it needs millions of tokens (App. scaling), the point.
Run with CUDA_VISIBLE_DEVICES=<free gpu> (GTR inverter needs a GPU; vLLM is on 2,3).
"""
import os, json, numpy as np
os.environ["OPENAI_API_KEY"] = "BLOCKED-cache-only"
from pathlib import Path
from sklearn.mixture import GaussianMixture
import common as C
from e3_gtr import invert_gtr

DATASETS = ["trump"]
NS = [1000, 10000]
SEEDS = [42, 43, 44]
K = 10
CACHE = Path("cache")
OUT = Path("results/gtr_quality"); OUT.mkdir(parents=True, exist_ok=True)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


for ds in DATASETS:
    for s in SEEDS:
        emb10k = np.load(CACHE / f"gtr_{ds}_{s}_10000.npy").astype(np.float32)
        docs10k = C.load_corpus(ds, 10000, s)
        for n in NS:
            f = OUT / f"{ds}_{n}_{s}.json"
            if f.exists():
                print(f"skip {f.name}"); continue
            emb, docs = emb10k[:n], docs10k[:n]
            eu = unit(emb)
            gmm = GaussianMixture(K, covariance_type="diag", random_state=s).fit(emb)
            modes = gmm.means_.astype(np.float32)
            modes_u = unit(modes)
            rng = np.random.default_rng(s)
            med_idx = (modes_u @ eu.T).argmax(1)
            cmean = unit(emb.mean(0, keepdims=True))
            cent_idx = np.argsort(-(eu @ cmean.T).ravel())[:K]
            rand_idx = rng.choice(n, K, replace=False)
            snippets = invert_gtr(modes)                      # GTR-invert the k modes
            texts = {
                "vec2summ_gmm": snippets,
                "medoid": [docs[i] for i in med_idx],
                "centroid": [docs[i] for i in cent_idx],
                "random": [docs[i] for i in rand_idx],
            }
            summaries = {c: C.summarize(t, local=True, model="llama-3.3-70b")
                         for c, t in texts.items()}
            json.dump({"dataset": ds, "n": n, "seed": s, "summaries": summaries,
                       "inverted_snippets": snippets,
                       "corpus_sample": docs[:120]}, open(f, "w"), indent=1)
            print(f"{ds} n={n} seed={s} done "
                  f"({sum(len(v) for v in summaries.values() if v)} summary chars)")
print("ALL DONE")
