"""Score the expert-panel human annotations against _truth.json.
Reads every human_eval/human_annotations_*.json (each {annotator, annotations:{cell:...}}).
Reports, per annotator and pooled: the Vec2Summ-vs-medoid coverage verdict + sign
test, per-system faithfulness rate, agreement with the silver draft (Cohen kappa),
and, with >=2 annotators, inter-annotator Fleiss kappa on the coverage judgment.
"""
import json, glob, math
from pathlib import Path
from collections import defaultdict

HE = Path("human_eval")
truth = json.load(open(HE / "_truth.json"))
silver = {p.stem: json.load(open(p)) for p in (HE / "silver").glob("*.json")}
files = sorted(glob.glob(str(HE / "human_annotations_*.json")))
if not files:
    raise SystemExit("no human_eval/human_annotations_*.json found yet")


def winner_system(cell, cw):
    return truth[cell].get(cw, "tie") if cw in ("S1", "S2") else "tie"


def sign_test(wins, n):                       # two-sided exact binomial, p=.5
    from math import comb
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def cohen_kappa(a, b):                         # two raters, categorical
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def fleiss_kappa(table):                       # rows=items, cols=category counts
    n_items = len(table); n_raters = sum(table[0])
    cats = len(table[0])
    p_j = [sum(row[j] for row in table) / (n_items * n_raters) for j in range(cats)]
    P_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1)) for row in table]
    Pbar = sum(P_i) / n_items; Pe = sum(p * p for p in p_j)
    return (Pbar - Pe) / (1 - Pe) if Pe != 1 else 1.0


anns = {}
for f in files:
    d = json.load(open(f))
    anns[d.get("annotator", Path(f).stem)] = d["annotations"]

cells = sorted({c for a in anns.values() for c in a})
print(f"=== {len(anns)} annotator(s): {', '.join(anns)} | {len(cells)} cells ===\n")

# per-annotator coverage verdict + faithfulness, and agreement with silver
for who, ann in anns.items():
    cov = defaultdict(int); fr = {"vec2summ_gmm": [], "medoid": []}
    hk, sk = [], []
    for c in cells:
        if c not in ann:
            continue
        ws = winner_system(c, ann[c]["coverage_winner"]); cov[ws] += 1
        for slot in ("S1", "S2"):
            fr[truth[c][slot]].append(ann[c][f"{slot}_faithfulness"]["faithfulness_rate"])
        if c in silver:
            hk.append(winner_system(c, ann[c]["coverage_winner"]))
            sk.append(winner_system(c, silver[c]["coverage_winner"]))
    nm = cov["medoid"]; ng = cov["vec2summ_gmm"]; nt = cov["tie"]
    p = sign_test(nm, nm + ng)
    print(f"[{who}] coverage: medoid {nm}, Vec2Summ {ng}, tie {nt}  (sign test p={p:.3f})")
    for s in ("vec2summ_gmm", "medoid"):
        print(f"        faithfulness {s:13}: {sum(fr[s])/len(fr[s]):.3f}")
    if sk:
        print(f"        agreement with silver (Cohen kappa, coverage): {cohen_kappa(hk, sk):.3f}")
    print()

# inter-annotator agreement on coverage winner (>=2 annotators)
if len(anns) >= 2:
    CATS = ["medoid", "vec2summ_gmm", "tie"]
    table = []
    for c in cells:
        row = [0, 0, 0]
        for ann in anns.values():
            if c in ann:
                row[CATS.index(winner_system(c, ann[c]["coverage_winner"]))] += 1
        if sum(row) == len(anns):
            table.append(row)
    print(f"Inter-annotator Fleiss kappa (coverage winner, {len(table)} cells): "
          f"{fleiss_kappa(table):.3f}")
    # pooled majority verdict
    maj_m = sum(1 for r in table if r[0] > max(r[1], r[2]))
    maj_g = sum(1 for r in table if r[1] > max(r[0], r[2]))
    print(f"Majority verdict: medoid {maj_m}, Vec2Summ {maj_g}, "
          f"split/tie {len(table)-maj_m-maj_g} (sign test p={sign_test(maj_m, maj_m+maj_g):.3f})")
