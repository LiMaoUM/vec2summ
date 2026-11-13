"""E1 + E2: cost-matched baselines and sampling ablations.

Per (dataset, n, seed) cell:
  conditions:
    vec2summ   - Gaussian sample (T=1.2, paper setting) -> vec2text inversion -> k snippets
    centroid   - the k actual documents nearest the corpus mean
    random     - k random documents
    direct     - all n documents (cost-unmatched upper bound)
  E2 sweep (subset of cells): vec2summ at T in {0, 0.5, 1.0, 1.5, 2.0}

Artifacts: results/e1/{ds}_{n}_{seed}.json   (snippets, recon metrics, summaries)
           results/e2/{ds}_{n}_{seed}_T{t}.json
Resumable: existing artifacts are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS = C.EXP_DIR / "results"
K = 10
PAPER_T = 1.2
SWEEP_T = [0.0, 0.5, 1.0, 1.5, 2.0]
SWEEP_CELLS = [("trump", 1000, 42), ("onebook", 1000, 42), ("askdocs", 1000, 42)]

_corrector = None


def get_corrector():
    global _corrector
    if _corrector is None:
        import vec2text
        _corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
    return _corrector


def invert(vectors, batch=2):
    """Mini-batched inversion: keeps beam-search peak memory small enough to
    coexist with the vLLM server on the same GPU."""
    import vec2text
    arr = np.asarray(vectors, dtype=np.float32)
    out = []
    for i in range(0, len(arr), batch):
        emb = torch.tensor(arr[i:i + batch]).cuda()
        out.extend(vec2text.invert_embeddings(
            embeddings=emb, corrector=get_corrector(),
            num_steps=5, sequence_beam_width=4))
        del emb
        torch.cuda.empty_cache()
    return out


def local_up():
    try:
        C._local_client.models.list()
        return True
    except Exception:
        return False


def summaries_for(texts, n_ctx_docs, use_local):
    out = {"gpt-4.1": C.summarize(texts, local=False)}
    out["llama-3.3-70b"] = C.summarize(texts, local=True) if use_local else None
    return out


def run_cell(ds, n, seed, use_local):
    out_f = RESULTS / "e1" / f"{ds}_{n}_{seed}.json"
    if out_f.exists():
        return "skip"
    corpus = C.load_corpus(ds, n, seed)
    emb = C.embed_openai(corpus, cache_tag=f"{ds}_{n}_{seed}")
    mu, cov = C.fit_gaussian(emb)
    rng = np.random.default_rng(seed)

    cell = {"dataset": ds, "n": n, "seed": seed, "k": K, "conditions": {}}

    # --- vec2summ (paper setting) ---
    vecs = C.sample_gaussian(mu, cov, K, PAPER_T, seed)
    recon_texts = invert(vecs)
    recon_emb = C.embed_openai(recon_texts)
    cell["conditions"]["vec2summ"] = {
        "texts": recon_texts,
        "recon": C.recon_quality(emb, recon_emb),
        "summaries": summaries_for(recon_texts, n, use_local),
    }

    # --- centroid-k ---
    sims_to_mu = (emb @ mu) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(mu) + 1e-9)
    cent_idx = np.argsort(-sims_to_mu)[:K]
    cent_texts = [corpus[i] for i in cent_idx]
    cell["conditions"]["centroid"] = {
        "texts": cent_texts,
        "recon": C.recon_quality(emb, emb[cent_idx]),
        "summaries": summaries_for(cent_texts, n, use_local),
    }

    # --- random-k ---
    rand_idx = rng.choice(n, K, replace=False)
    rand_texts = [corpus[i] for i in rand_idx]
    cell["conditions"]["random"] = {
        "texts": rand_texts,
        "recon": C.recon_quality(emb, emb[rand_idx]),
        "summaries": summaries_for(rand_texts, n, use_local),
    }

    # --- direct (full corpus in context) ---
    cell["conditions"]["direct"] = {
        "texts": None,
        "recon": None,
        "summaries": summaries_for(corpus, n, use_local),
    }

    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(cell, indent=1))
    return "done"


def run_sweep_cell(ds, n, seed, t, use_local):
    out_f = RESULTS / "e2" / f"{ds}_{n}_{seed}_T{t}.json"
    if out_f.exists():
        return "skip"
    corpus = C.load_corpus(ds, n, seed)
    emb = C.embed_openai(corpus, cache_tag=f"{ds}_{n}_{seed}")
    mu, cov = C.fit_gaussian(emb)
    recon_texts = invert(C.sample_gaussian(mu, cov, K, t, seed))
    recon_emb = C.embed_openai(recon_texts)
    res = {
        "dataset": ds, "n": n, "seed": seed, "T": t, "k": K,
        "texts": recon_texts,
        "recon": C.recon_quality(emb, recon_emb),
        "summaries": summaries_for(recon_texts, n, use_local),
    }
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(res, indent=1))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    ap.add_argument("--sizes", nargs="*", type=int, default=[100, 1000])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    use_local = local_up()
    print(f"[e1] local llama server up: {use_local}", flush=True)

    for ds in args.datasets:
        for n in args.sizes:
            for seed in args.seeds:
                try:
                    status = run_cell(ds, n, seed, use_local)
                    print(f"[e1] {ds} n={n} seed={seed}: {status}", flush=True)
                except Exception as e:
                    print(f"[e1] {ds} n={n} seed={seed}: ERROR {e!r}", flush=True)

    if not args.skip_sweep:
        for ds, n, seed in SWEEP_CELLS:
            for t in SWEEP_T:
                try:
                    status = run_sweep_cell(ds, n, seed, t, use_local)
                    print(f"[e2] {ds} n={n} T={t}: {status}", flush=True)
                except Exception as e:
                    print(f"[e2] {ds} n={n} T={t}: ERROR {e!r}", flush=True)


if __name__ == "__main__":
    main()
