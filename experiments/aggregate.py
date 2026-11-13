"""Aggregate all experiment artifacts into paper-ready tables (CSV + Markdown).

Outputs to results/tables/:
  recon_quality.csv      E1a: mean_max_sim per condition x dataset x n (mean over seeds)
  geval_scores.csv       E6: per-criterion judge scores per condition x summarizer
  temp_sweep.csv         E2: recon + scores vs temperature
  hierarchical.csv       E7: map-reduce quality placeholder join + token cost
  cost_table.csv         E4: approx tokens through the summarizer LLM per condition
  SUMMARY.md             headline findings
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS = C.EXP_DIR / "results"
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def approx_tokens(s):
    return len(s) // 4 if s else 0


def load_dir(d):
    return [json.loads(p.read_text()) for p in sorted((RESULTS / d).glob("*.json"))]


def recon_table():
    rows = []
    for cell in load_dir("e1"):
        for cond, blob in cell["conditions"].items():
            if blob.get("recon"):
                rows.append({"dataset": cell["dataset"], "n": cell["n"],
                             "seed": cell["seed"], "condition": cond,
                             **blob["recon"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["dataset", "n"], columns="condition",
                         values="mean_max_sim", aggfunc="mean").round(4)
    if {"vec2summ", "random"} <= set(piv.columns):
        piv["v2s_minus_random"] = (piv["vec2summ"] - piv["random"]).round(4)
    if {"vec2summ", "centroid"} <= set(piv.columns):
        piv["v2s_minus_centroid"] = (piv["vec2summ"] - piv["centroid"]).round(4)
    piv.to_csv(TABLES / "recon_quality.csv")
    return piv


def geval_table():
    rows = []
    for g in load_dir("geval"):
        for cond, by_sum in g["scores"].items():
            for summarizer, crits in by_sum.items():
                row = {"dataset": g["dataset"], "n": g["n"], "seed": g["seed"],
                       "T": g.get("T"), "condition": cond, "summarizer": summarizer}
                for crit, blob in crits.items():
                    row[crit] = blob.get("score")
                rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "geval_scores_raw.csv", index=False)
    crits = ["coverage", "factual_accuracy", "coherence", "conciseness"]
    main = df[df["T"].isna()] if df["T"].notna().any() else df
    piv = main.pivot_table(index=["condition", "summarizer"],
                           values=[c for c in crits if c in df], aggfunc="mean").round(3)
    piv.to_csv(TABLES / "geval_scores.csv")
    return piv


def sweep_table():
    rows = []
    for cell in load_dir("e2"):
        rows.append({"dataset": cell["dataset"], "T": cell["T"],
                     **cell["recon"]})
    if not rows:
        return None
    df = pd.DataFrame(rows).pivot_table(index="T", columns="dataset",
                                        values="mean_max_sim").round(4)
    df.to_csv(TABLES / "temp_sweep.csv")
    return df


def cost_table():
    rows = []
    for cell in load_dir("e1"):
        ds, n, seed = cell["dataset"], cell["n"], cell["seed"]
        corpus = C.load_corpus(ds, n, seed)
        corpus_toks = sum(approx_tokens(t[:1500]) for t in corpus)
        for cond, blob in cell["conditions"].items():
            if cond == "direct":
                toks = corpus_toks
            else:
                toks = sum(approx_tokens(t[:1500]) for t in (blob.get("texts") or []))
            rows.append({"dataset": ds, "n": n, "seed": seed, "condition": cond,
                         "summarizer_input_tokens": toks})
    for h in load_dir("e7"):
        rows.append({"dataset": h["dataset"], "n": h["n"], "seed": h["seed"],
                     "condition": f"hierarchical[{h['summarizer']}]",
                     "summarizer_input_tokens": h["approx_input_tokens"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["dataset", "n"], columns="condition",
                         values="summarizer_input_tokens", aggfunc="mean").round(0)
    piv.to_csv(TABLES / "cost_table.csv")
    return piv


def e8_table():
    rows = []
    for a in load_dir("e8"):
        for cond, audit in a["audits"].items():
            if audit["unsupported_rate"] is not None:
                rows.append({"dataset": a["dataset"], "n": a["n"], "seed": a["seed"],
                             "condition": cond,
                             "unsupported_rate": audit["unsupported_rate"],
                             "n_claims": audit["n_claims"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="condition",
                         values=["unsupported_rate", "n_claims"], aggfunc="mean").round(3)
    piv.to_csv(TABLES / "factuality.csv")
    return piv


def main():
    parts = ["# Revision experiment summary\n"]
    for name, fn in [("E1a Reconstruction quality (mean_max_sim)", recon_table),
                     ("E6 Per-criterion G-Eval", geval_table),
                     ("E2 Temperature sweep", sweep_table),
                     ("E4 Cost (approx summarizer input tokens)", cost_table),
                     ("E8 Factuality (unsupported-claim rate)", e8_table)]:
        t = fn()
        parts.append(f"## {name}\n")
        parts.append(t.to_markdown() + "\n" if t is not None else "_no data yet_\n")
    (TABLES / "SUMMARY.md").write_text("\n".join(parts))
    print(f"tables written to {TABLES}")


if __name__ == "__main__":
    main()
