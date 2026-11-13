"""E6: per-criterion G-Eval over all e1/e2 artifacts.

For every (cell, condition, summarizer) summary, judge coverage / factual
accuracy / coherence / conciseness with separate prompts.
The judge sees the same seeded source subsample for every condition in a
cell, so scores are comparable within-cell.

Output: results/geval/{artifact-stem}.json . Resumable.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS = C.EXP_DIR / "results"
OUT = RESULTS / "geval"


def eval_artifact(path):
    out_f = OUT / path.name
    if out_f.exists():
        return "skip"
    cell = json.loads(path.read_text())
    ds, n, seed = cell["dataset"], cell["n"], cell["seed"]
    corpus = C.load_corpus(ds, n, seed)

    conditions = cell.get("conditions") or {"vec2summ_sweep": cell}
    scores = {}
    for cond, blob in conditions.items():
        scores[cond] = {}
        for summarizer, summary in (blob.get("summaries") or {}).items():
            if not summary:
                continue
            scores[cond][summarizer] = C.geval(summary, corpus, seed=seed)
    res = {"dataset": ds, "n": n, "seed": seed,
           "T": cell.get("T"), "scores": scores}
    OUT.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(res, indent=1))
    return "done"


def main():
    arts = (sorted((RESULTS / "e1").glob("*.json"))
            + sorted((RESULTS / "e2").glob("*.json"))
            + sorted((RESULTS / "e9").glob("*.json")))
    print(f"[geval] {len(arts)} artifacts", flush=True)
    for p in arts:
        try:
            print(f"[geval] {p.name}: {eval_artifact(p)}", flush=True)
        except Exception as e:
            print(f"[geval] {p.name}: ERROR {e!r}", flush=True)


if __name__ == "__main__":
    main()
