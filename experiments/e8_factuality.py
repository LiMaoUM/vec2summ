"""E8: claim-level factuality audit (ungrounded Gaussian sampling may
produce false reconstructions).

For each e1 artifact and condition summary (gpt-4.1 summarizer):
  1. extract atomic factual claims from the summary (LLM, JSON)
  2. for each claim, retrieve top-RETRIEVE corpus docs by ada-002 cosine
  3. judge SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED given the evidence

Hallucination rate = fraction UNSUPPORTED. Vec2Summ vs grounded baselines
(random/centroid/direct) is the comparison of interest.

Artifacts: results/e8/{stem}.json . Resumable.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS = C.EXP_DIR / "results"
RETRIEVE = 8

EXTRACT_PROMPT = (
    "Extract the distinct factual claims made in this summary of a text "
    "collection. A claim is a single checkable statement about the collection's "
    "content (topics, opinions expressed, entities, sentiments). Ignore "
    "meta-statements about style or structure. Return JSON: "
    '{{"claims": ["...", ...]}} (max 12 claims).\n\nSummary:\n{summary}'
)

VERIFY_PROMPT = (
    "You are verifying whether a claim about a text collection is supported "
    "by evidence from that collection.\n\nClaim: {claim}\n\nEvidence (most "
    "similar documents from the collection):\n{evidence}\n\nJudge: SUPPORTED "
    "(evidence clearly backs the claim), PARTIALLY_SUPPORTED (related but not "
    "clearly backing), or UNSUPPORTED (no evidence or contradicted).\n"
    'Reply JSON: {{"verdict": "...", "rationale": "..."}}'
)


def parse_json(reply, default):
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        return json.loads(m.group(0))
    except Exception:
        return default


def audit_summary(summary, corpus, corpus_emb):
    claims = parse_json(
        C.chat([{"role": "user", "content": EXTRACT_PROMPT.format(summary=summary)}],
               temperature=0.0, max_tokens=900),
        {"claims": []})["claims"][:12]
    results = []
    if claims:
        claim_emb = C.embed_openai(claims)
        norm_c = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)
        norm_q = claim_emb / (np.linalg.norm(claim_emb, axis=1, keepdims=True) + 1e-9)
        sims = norm_q @ norm_c.T
        for ci, claim in enumerate(claims):
            top = np.argsort(-sims[ci])[:RETRIEVE]
            evidence = "\n".join(f"- [{corpus[i][:600]}]" for i in top)
            v = parse_json(
                C.chat([{"role": "user", "content": VERIFY_PROMPT.format(
                    claim=claim, evidence=evidence)}], temperature=0.0, max_tokens=400),
                {"verdict": "PARSE_ERROR"})
            results.append({"claim": claim, "verdict": v.get("verdict"),
                            "rationale": v.get("rationale")})
    n_uns = sum(1 for r in results if r["verdict"] == "UNSUPPORTED")
    return {"claims": results,
            "n_claims": len(results),
            "unsupported_rate": (n_uns / len(results)) if results else None}


def main():
    out_dir = RESULTS / "e8"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (sorted((RESULTS / "e1").glob("*.json"))
                 + sorted((RESULTS / "e9").glob("*.json"))):
        out_f = out_dir / path.name
        if out_f.exists():
            print(f"[e8] {path.name}: skip", flush=True)
            continue
        try:
            cell = json.loads(path.read_text())
            ds, n, seed = cell["dataset"], cell["n"], cell["seed"]
            corpus = C.load_corpus(ds, n, seed)
            emb = C.embed_openai(corpus, cache_tag=f"{ds}_{n}_{seed}")
            res = {"dataset": ds, "n": n, "seed": seed, "audits": {}}
            for cond, blob in cell["conditions"].items():
                s = (blob.get("summaries") or {}).get("gpt-4.1")
                if s:
                    res["audits"][cond] = audit_summary(s, corpus, emb)
            out_f.write_text(json.dumps(res, indent=1))
            print(f"[e8] {path.name}: done", flush=True)
        except Exception as e:
            print(f"[e8] {path.name}: ERROR {e!r}", flush=True)


if __name__ == "__main__":
    main()
