"""Build blinded judge packets for the GTR scale-quality run.
Each cell -> one packet: a 120-doc corpus sample + the 4 condition summaries
relabeled A-D in a per-cell shuffled order. Keymap saved for unblinding.
"""
import json, random
from pathlib import Path

SRC = Path("results/gtr_quality")
OUT = Path("results/gtr_packets"); OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["vec2summ_gmm", "medoid", "centroid", "random"]
keymap = {}

for f in sorted(SRC.glob("*.json")):
    d = json.load(open(f))
    cell = f.stem                                  # e.g. trump_10000_42
    rng = random.Random(hash(cell) & 0xffffffff)
    order = CONDS[:]; rng.shuffle(order)
    letters = ["A", "B", "C", "D"]
    keymap[cell] = {letters[i]: order[i] for i in range(4)}
    sample = "\n".join(f"- {t[:300]}" for t in d["corpus_sample"])
    summaries = {letters[i]: d["summaries"][order[i]] for i in range(4)}
    json.dump({"cell": cell, "n": d["n"], "corpus_sample": sample,
               "summaries": summaries}, open(OUT / f"{cell}.json", "w"), indent=1)

json.dump(keymap, open(OUT / "_keymap.json", "w"), indent=1)
print(f"built {len(keymap)} packets -> {OUT}")
print("conditions blinded as A-D per cell; keymap saved.")
