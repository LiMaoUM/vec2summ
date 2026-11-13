"""Build a human-evaluation subset (SPACE/FRANK-style) for the pivotal
Vec2Summ(gmm) vs medoid-k coverage comparison. Reuses the n=1000 medoid-run
packets (results/codex_packets). For each selected cell we emit one item with:
 - source_sample: the 120-doc corpus sample (the salience grounding)
 - summary_S1 / summary_S2: the two systems, RE-BLINDED (random S1/S2 order)
 - per-summary sentence split (for FRANK-style faithfulness)
Writes human_eval/items/*.json + human_eval/_truth.json (S1/S2 -> system map).
A model-assisted silver pass (Opus) then annotates these; a human validates.
"""
import json, re, random
from pathlib import Path

SRC = Path("results/codex_packets")
OUT = Path("human_eval"); (OUT / "items").mkdir(parents=True, exist_ok=True)
keymap = json.load(open(SRC / "_keymap.json"))

# 3 domains x 3 seeds (matches the scaling corpora: health / tweets / narratives)
DATASETS = ["askdocs", "trump", "tifu"]
SEEDS = [42, 43, 44]
PAIR = ("vec2summ_gmm", "medoid")


def sentences(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in parts if len(s.strip()) > 3]


truth = {}
for ds in DATASETS:
    for s in SEEDS:
        cell = f"{ds}_{s}"
        pkt = json.load(open(SRC / f"{cell}.json"))
        letter = {v: k for k, v in keymap[cell].items()}  # condition -> letter
        sums = {sys: pkt["summaries"][letter[sys]] for sys in PAIR}
        rng = random.Random(hash(cell) & 0xffffffff)
        order = list(PAIR); rng.shuffle(order)            # re-blind S1/S2
        truth[cell] = {"S1": order[0], "S2": order[1]}
        item = {
            "cell": cell, "dataset": ds, "seed": s,
            "n_source_shown": len(pkt["source_docs"]),
            "source_sample": pkt["source_docs"],
            "summary_S1": sums[order[0]], "summary_S2": sums[order[1]],
            "S1_sentences": sentences(sums[order[0]]),
            "S2_sentences": sentences(sums[order[1]]),
        }
        json.dump(item, open(OUT / "items" / f"{cell}.json", "w"), indent=1)

json.dump(truth, open(OUT / "_truth.json", "w"), indent=1)
print(f"built {len(truth)} human-eval items in {OUT/'items'} (blinded S1/S2)")
print("systems compared:", PAIR, "| truth map (cell->S1/S2 system) in _truth.json")
for c, t in list(truth.items())[:3]:
    print(" ", c, t)
