"""Build cross-judge packets: reconstruct the exact 120-doc sample each GPT-4.1
G-Eval cell saw, pair it with the actual summaries, write one packet per
(dataset, seed). Judges (Claude subagents) read these and score per the same
rubric used by the original GPT-4.1 judge (common.GEVAL_PROMPTS / geval()).
"""
import os, json
os.environ.setdefault("OPENAI_API_KEY", "dummy")
import numpy as np
import common as C

DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
# condition key -> (artifact dir, file suffix, condition name inside JSON)
COND = {
    "direct":        ("e1", "",      "direct"),
    "centroid":      ("e1", "",      "centroid"),
    "random":        ("e1", "",      "random"),
    "vec2summ_gmm":  ("e9", "_gmm",  "vec2summ_gmm"),
    "vec2summ_pca":  ("e9", "_pca",  "vec2summ_pca"),
    "gauss":         ("e1", "",      "vec2summ"),   # Gaussian-sampling ablation
}
RES = C.EXP_DIR / "results"
OUT = RES / "judge_packets"
OUT.mkdir(exist_ok=True)


def source_sample(ds, seed, cap=120):
    """Exactly reproduce common.geval's seeded subsample (rng=seed, 800-char cap)."""
    docs = C.load_corpus(ds, 1000, seed)
    rng = np.random.default_rng(seed)
    if len(docs) > cap:
        idx = rng.choice(len(docs), cap, replace=False)
        docs = [docs[i] for i in sorted(idx)]
    return [d[:800] for d in docs]


def summary_for(ds, seed, cond):
    d, suf, name = COND[cond]
    p = RES / d / f"{ds}_1000_{seed}{suf}.json"
    j = json.load(open(p))
    return j["conditions"][name]["summaries"]["gpt-4.1"]


# Blinding: each packet presents summaries under opaque IDs in a fixed,
# seed-derived shuffle, so the judge cannot tell which is the proposed method.
# The true mapping is kept out of the judge's packet, in _keymap.json.
LABELS = ["S1", "S2", "S3", "S4", "S5", "S6"]
keymap = {}
for ds in DATASETS:
    for seed in SEEDS:
        src = source_sample(ds, seed)
        conds = list(COND)
        order = np.random.default_rng(1000 + seed * 7 + hash(ds) % 100).permutation(len(conds))
        blind = {LABELS[i]: summary_for(ds, seed, conds[order[i]]) for i in range(len(conds))}
        mapping = {LABELS[i]: conds[order[i]] for i in range(len(conds))}
        packet = {"dataset": ds, "seed": seed, "source_docs": src, "summaries": blind}
        json.dump(packet, open(OUT / f"{ds}_{seed}.json", "w"))
        keymap[f"{ds}_{seed}"] = mapping
json.dump(keymap, open(OUT / "_keymap.json", "w"), indent=1)
print(f"wrote {len(DATASETS)*len(SEEDS)} blinded packets to {OUT}")
# size sanity
import glob
sizes = [os.path.getsize(f) for f in glob.glob(str(OUT / "*.json"))]
print(f"packet size: min {min(sizes)//1024}KB max {max(sizes)//1024}KB")
