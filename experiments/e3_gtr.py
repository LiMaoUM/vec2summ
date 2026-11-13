"""E3: embedding-model ablation — GTR (gtr-t5-base) pipeline, fully local.

Mirrors the e1 conditions in GTR space. Cross-space absolute similarities are
not comparable; the meaningful quantity is the WITHIN-SPACE lift of vec2summ
over random/centroid, compared with the same lift in ada-002 space (from e1).

Subset: all datasets, n=1000, seed=42. Summaries: gpt-4.1 only.
Artifacts: results/e3/{ds}_{n}_{seed}.json . Resumable.
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

_models = None


def get_models():
    global _models
    if _models is None:
        import vec2text
        from transformers import AutoModel, AutoTokenizer
        enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.cuda()
        tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        corr = vec2text.load_pretrained_corrector("gtr-base")
        _models = (enc, tok, corr)
    return _models


def embed_gtr(texts, batch_size=64):
    import vec2text
    enc, tok, _ = get_models()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:2000] for t in texts[i:i + batch_size]]
        inputs = tok(batch, return_tensors="pt", max_length=128,
                     truncation=True, padding="longest").to("cuda")
        with torch.no_grad():
            hidden = enc(input_ids=inputs.input_ids,
                         attention_mask=inputs.attention_mask).last_hidden_state
            emb = vec2text.models.model_utils.mean_pool(hidden, inputs.attention_mask)
        out.append(emb.cpu().float())
        torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def invert_gtr(vectors, batch=2):
    import vec2text
    _, _, corr = get_models()
    arr = np.asarray(vectors, dtype=np.float32)
    out = []
    for i in range(0, len(arr), batch):
        emb = torch.tensor(arr[i:i + batch]).cuda()
        out.extend(vec2text.invert_embeddings(
            embeddings=emb, corrector=corr, num_steps=5, sequence_beam_width=4))
        del emb
        torch.cuda.empty_cache()
    return out


def run_cell(ds, n, seed):
    out_f = RESULTS / "e3" / f"{ds}_{n}_{seed}.json"
    if out_f.exists():
        return "skip"
    corpus = C.load_corpus(ds, n, seed)
    emb = embed_gtr(corpus)
    mu, cov = C.fit_gaussian(emb)
    rng = np.random.default_rng(seed)

    cell = {"dataset": ds, "n": n, "seed": seed, "k": K,
            "embedder": "gtr-t5-base", "conditions": {}}

    recon_texts = invert_gtr(C.sample_gaussian(mu, cov, K, PAPER_T, seed))
    cell["conditions"]["vec2summ"] = {
        "texts": recon_texts,
        "recon": C.recon_quality(emb, embed_gtr(recon_texts)),
        "summaries": {"gpt-4.1": C.summarize(recon_texts)},
    }

    sims_to_mu = (emb @ mu) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(mu) + 1e-9)
    cent_idx = np.argsort(-sims_to_mu)[:K]
    cell["conditions"]["centroid"] = {
        "texts": [corpus[i] for i in cent_idx],
        "recon": C.recon_quality(emb, emb[cent_idx]),
        "summaries": {"gpt-4.1": C.summarize([corpus[i] for i in cent_idx])},
    }

    rand_idx = rng.choice(n, K, replace=False)
    cell["conditions"]["random"] = {
        "texts": [corpus[i] for i in rand_idx],
        "recon": C.recon_quality(emb, emb[rand_idx]),
        "summaries": {"gpt-4.1": C.summarize([corpus[i] for i in rand_idx])},
    }

    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(cell, indent=1))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    ap.add_argument("--sizes", nargs="*", type=int, default=[1000])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = ap.parse_args()
    for ds in args.datasets:
        for n in args.sizes:
            for seed in args.seeds:
                try:
                    print(f"[e3] {ds} n={n} seed={seed}: {run_cell(ds, n, seed)}", flush=True)
                except Exception as e:
                    print(f"[e3] {ds} n={n} seed={seed}: ERROR {e!r}", flush=True)


if __name__ == "__main__":
    main()
