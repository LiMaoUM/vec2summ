"""E7: hierarchical map-reduce LLM summarization baseline.

Chunk the corpus into groups of CHUNK docs, summarize each chunk, then
summarize the chunk-summaries (single reduce level; two levels cover
n <= CHUNK^2). Token usage is tracked so cost can be reported jointly
with quality (the point of the comparison vs Vec2Summ's O(k)).

Artifacts: results/e7/{ds}_{n}_{seed}_{summarizer}.json . Resumable.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS = C.EXP_DIR / "results"
CHUNK = 50

CHUNK_PROMPT = (
    "Summarize the following texts in ONE paragraph of roughly 120 words, "
    "capturing main points and overall sentiment.\n\nTexts:\n{texts}\n\nSummary:"
)
REDUCE_PROMPT = (
    "The following are summaries of consecutive batches of a large text "
    "collection. Combine them into a single-paragraph summary of roughly "
    "150 words capturing the main points and overall sentiment of the whole "
    "collection.\n\nBatch summaries:\n{texts}\n\nCombined summary:"
)


def approx_tokens(s):
    return len(s) // 4


def map_reduce(corpus, local):
    toks_in = 0
    chunk_summaries = []
    for i in range(0, len(corpus), CHUNK):
        chunk = corpus[i:i + CHUNK]
        body = "\n".join(f"{j+1}. [{t[:1500]}]" for j, t in enumerate(chunk))
        toks_in += approx_tokens(body)
        chunk_summaries.append(C.chat(
            [{"role": "user", "content": CHUNK_PROMPT.format(texts=body)}], local=local))
    body = "\n\n".join(f"Batch {j+1}: {s}" for j, s in enumerate(chunk_summaries))
    toks_in += approx_tokens(body)
    final = C.chat([{"role": "user", "content": REDUCE_PROMPT.format(texts=body)}], local=local)
    return final, chunk_summaries, toks_in


def run(ds, n, seed, summarizer):
    out_f = RESULTS / "e7" / f"{ds}_{n}_{seed}_{summarizer}.json"
    if out_f.exists():
        return "skip"
    corpus = C.load_corpus(ds, n, seed)
    final, chunks, toks = map_reduce(corpus, local=(summarizer != "gpt-4.1"))
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps({
        "dataset": ds, "n": n, "seed": seed, "summarizer": summarizer,
        "n_chunks": len(chunks), "approx_input_tokens": toks,
        "summary": final, "chunk_summaries": chunks}, indent=1))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(C.DATASETS))
    ap.add_argument("--sizes", nargs="*", type=int, default=[1000])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--summarizers", nargs="*", default=["gpt-4.1", "llama-3.3-70b"])
    args = ap.parse_args()
    for ds in args.datasets:
        for n in args.sizes:
            for seed in args.seeds:
                for sm in args.summarizers:
                    try:
                        print(f"[e7] {ds} n={n} seed={seed} {sm}: {run(ds, n, seed, sm)}",
                              flush=True)
                    except Exception as e:
                        print(f"[e7] {ds} n={n} seed={seed} {sm}: ERROR {e!r}", flush=True)


if __name__ == "__main__":
    main()
