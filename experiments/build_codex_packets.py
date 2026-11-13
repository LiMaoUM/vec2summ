"""Blinded judge packets for the codex evaluation, Llama-summarizer set.
Reuses the exact 120-doc source samples from judge_packets/, swaps in the
Llama-3.3-70B summaries of each condition plus the new medoid-k summary, and
blinds them behind opaque IDs. One packet per (dataset, seed).
"""
import os, json
os.environ["OPENAI_API_KEY"] = "dummy"
import numpy as np
import common as C

RES = C.EXP_DIR / "results"
DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
SEEDS = [42, 43, 44]
OUT = RES / "codex_packets"
OUT.mkdir(exist_ok=True)
LL = "llama-3.3-70b"

def llama_summ(ds, s, cond):
    if cond == "medoid":
        return json.load(open(RES/"e10_medoid"/f"{ds}_{s}.json"))["summary_llama"]
    if cond == "vec2summ_gmm":
        return json.load(open(RES/"e9"/f"{ds}_1000_{s}_gmm.json"))["conditions"]["vec2summ_gmm"]["summaries"][LL]
    return json.load(open(RES/"e1"/f"{ds}_1000_{s}.json"))["conditions"][cond]["summaries"][LL]

CONDS = ["medoid", "vec2summ_gmm", "centroid", "random", "direct"]
LABELS = ["A", "B", "C", "D", "E"]
keymap = {}
missing = []
for ds in DATASETS:
    for s in SEEDS:
        cell = f"{ds}_{s}"
        try:
            src = json.load(open(RES/"judge_packets"/f"{cell}.json"))["source_docs"]
        except FileNotFoundError:
            missing.append(cell); continue
        order = np.random.default_rng(500+s*13+(hash(ds) % 100)).permutation(len(CONDS))
        summ = {}
        mp = {}
        ok = True
        for i, oi in enumerate(order):
            try:
                summ[LABELS[i]] = llama_summ(ds, s, CONDS[oi])
            except (FileNotFoundError, KeyError):
                ok = False; break
            mp[LABELS[i]] = CONDS[oi]
        if not ok:
            missing.append(cell); continue
        json.dump({"dataset": ds, "seed": s, "source_docs": src, "summaries": summ},
                  open(OUT/f"{cell}.json", "w"))
        keymap[cell] = mp
json.dump(keymap, open(OUT/"_keymap.json", "w"), indent=1)
print(f"wrote {len(keymap)} codex packets; missing/skipped: {missing}")
