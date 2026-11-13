"""Shared utilities for the Vec2Summ experiments.

Corpus loading, cached embedding and LLM calls, the mean-max similarity
metric and the judge prompts shared by the experiment scripts.
"""

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EXP_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(os.environ.get("VEC2SUMM_DATA", EXP_DIR.parent / "data"))

# dataset name -> (path, format, text column)
DATASETS = {
    "trump": (DATA_DIR / "variability/trump.csv", "csv", "Message"),
    "biden": (DATA_DIR / "variability/biden.csv", "csv", "Message"),
    "illegal": (DATA_DIR / "variability/illegal.csv", "csv", "Message"),
    "askdocs_fever": (DATA_DIR / "askdocs_fever.csv", "csv", "Question"),
    "askdocs": (DATA_DIR / "AskDocs.csv", "csv", "Question"),
    "onebook": (DATA_DIR / "onebook.csv", "csv", "review_body"),
    "tifu": (DATA_DIR / "tifu_all_tokenized_and_filtered.json", "jsonl", "selftext_without_tldr"),
}

_client = openai.OpenAI()
_local_client = openai.OpenAI(base_url="http://localhost:8300/v1", api_key="local")
LOCAL_MODEL = "llama-3.3-70b"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^RT\s+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(name, n, seed):
    """Seeded sample of n cleaned documents (deduplicated, length-filtered)."""
    path, fmt, col = DATASETS[name]
    if fmt == "jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    texts = df[col].dropna().astype(str).map(clean_text)
    texts = texts[texts.str.len().between(20, 6000)].drop_duplicates()
    if len(texts) < n:
        raise ValueError(f"{name}: only {len(texts)} docs after filtering, need {n}")
    return texts.sample(n, random_state=seed).tolist()


def _cache_key(*parts):
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:16]


def embed_openai(texts, model="text-embedding-ada-002", cache_tag=None, batch_size=128):
    """Embeddings with optional disk cache (np.float32 [n, d])."""
    if cache_tag:
        f = CACHE_DIR / f"emb_{model}_{_cache_key(cache_tag, len(texts))}.npy"
        if f.exists():
            return np.load(f)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:20000] for t in texts[i:i + batch_size]]
        for attempt in range(5):
            try:
                resp = _client.embeddings.create(input=batch, model=model, encoding_format="float")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        out.extend(e.embedding for e in resp.data)
    arr = np.asarray(out, dtype=np.float32)
    if cache_tag:
        np.save(f, arr)
    return arr


def fit_gaussian(embeddings):
    mu = embeddings.mean(axis=0)
    cov = np.cov(embeddings, rowvar=False)
    min_eig = np.min(np.real(np.linalg.eigvalsh(cov)))
    if min_eig < 1e-6:
        cov += (1e-6 - min_eig + 1e-4) * np.eye(cov.shape[0])
    return mu, cov


def sample_gaussian(mu, cov, k, temperature, seed):
    rng = np.random.default_rng(seed)
    if temperature == 0:
        return np.tile(mu, (k, 1))
    return rng.multivariate_normal(mu, temperature * cov, size=k)


def recon_quality(corpus_emb, candidate_emb):
    """Paper metric: mean over corpus docs of max cosine sim to any candidate."""
    sims = cosine_similarity(corpus_emb, candidate_emb)
    per_doc_max = sims.max(axis=1)
    return {
        "mean_max_sim": float(per_doc_max.mean()),
        "median_max_sim": float(np.median(per_doc_max)),
        "min_max_sim": float(per_doc_max.min()),
        "std_max_sim": float(per_doc_max.std()),
    }


def chat(messages, model="gpt-4.1", temperature=0.7, max_tokens=1024, local=False):
    cl = _local_client if local else _client
    mdl = LOCAL_MODEL if local else model
    for attempt in range(5):
        try:
            r = cl.chat.completions.create(
                model=mdl, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


SUMMARY_PROMPT = (
    "You are given a collection of texts. Write a summary that captures the main "
    "points, key information, and overall sentiment of the collection.\n"
    "Constraints: the summary must be a SINGLE paragraph of roughly {words} words.\n\n"
    "Texts:\n{texts}\n\nSummary:"
)


# Local llama context budget (served max-model-len 131072; ~4 chars/token,
# generous margin for prompt scaffold + output).
LOCAL_CTX_CHARS = 460_000


def summarize(texts, words=150, local=False, model="gpt-4.1", per_doc_char_cap=1500):
    cap = per_doc_char_cap
    if local:
        budget = LOCAL_CTX_CHARS // max(len(texts), 1)
        cap = max(200, min(cap, budget))
    body = "\n".join(f"{i+1}. [{t[:cap]}]" for i, t in enumerate(texts))
    try:
        return chat(
            [{"role": "user", "content": SUMMARY_PROMPT.format(words=words, texts=body)}],
            model=model, local=local)
    except openai.BadRequestError:
        if not local:
            raise
        return None  # context overflow even after truncation; fill later if needed


# ---------------- per-criterion G-Eval (decomposed prompts) -------------

GEVAL_PROMPTS = {
    "coverage": (
        "You are an expert evaluator of multi-document summarization.\n"
        "Below is a sample of documents from a corpus, and a summary of the WHOLE corpus.\n"
        "Evaluate COVERAGE: how well the summary captures the important themes and "
        "information of the corpus.\n1 = misses almost everything, 5 = covers all or "
        "almost all important themes.\n\nDocuments:\n{source}\n\nSummary:\n{summary}\n\n"
        "Reply with JSON: {{\"score\": <1-5>, \"missing\": [\"...\"]}}"
    ),
    "factual_accuracy": (
        "You are an expert fact-checking evaluator.\n"
        "Below is a sample of documents from a corpus, and a summary of the WHOLE corpus.\n"
        "Evaluate FACTUAL ACCURACY: are the claims in the summary supported by the "
        "documents? Penalize claims that contradict or are absent from the documents.\n"
        "1 = mostly unsupported/hallucinated, 5 = fully supported.\n\nDocuments:\n{source}"
        "\n\nSummary:\n{summary}\n\nReply with JSON: {{\"score\": <1-5>, "
        "\"unsupported_claims\": [\"...\"]}}"
    ),
    "coherence": (
        "You are an expert evaluator of text quality.\nEvaluate the COHERENCE of the "
        "following summary: logical flow, organization, readability as a standalone text.\n"
        "1 = incoherent fragments, 5 = perfectly coherent.\n\nSummary:\n{summary}\n\n"
        "Reply with JSON: {{\"score\": <1-5>, \"issues\": [\"...\"]}}"
    ),
    "conciseness": (
        "You are an expert evaluator of text quality.\nEvaluate the CONCISENESS of the "
        "following summary: information density, absence of redundancy and filler.\n"
        "1 = bloated/redundant, 5 = maximally concise while informative.\n\nSummary:\n"
        "{summary}\n\nReply with JSON: {{\"score\": <1-5>, \"issues\": [\"...\"]}}"
    ),
}


def _parse_json_reply(reply):
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    return json.loads(m.group(0)) if m else {"score": None, "raw": reply}


def geval(summary, source_docs, judge_model="gpt-4.1", source_cap=120, seed=0):
    """Per-criterion judge. Source shown to judge is a seeded subsample (cap docs)."""
    rng = np.random.default_rng(seed)
    docs = list(source_docs)
    if len(docs) > source_cap:
        idx = rng.choice(len(docs), source_cap, replace=False)
        docs = [docs[i] for i in sorted(idx)]
    source = "\n".join(f"{i+1}. [{d[:800]}]" for i, d in enumerate(docs))
    out = {}
    for crit, tmpl in GEVAL_PROMPTS.items():
        reply = chat(
            [{"role": "user", "content": tmpl.format(source=source, summary=summary)}],
            model=judge_model, temperature=0.0, max_tokens=800)
        out[crit] = _parse_json_reply(reply)
    return out


def run_parallel(fn, items, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
