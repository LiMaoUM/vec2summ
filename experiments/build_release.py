"""Build the public release artifacts that carry no source text.

Cache-only: embeddings are read from cache/, and a cache miss raises.
Writes, under the directory given as argv[1]:
  tweet_ids/     tweet IDs of the three Twitter corpora and of each seeded sample
  sketches/      fitted GMM sketches after the nearest-document threshold
  snippets/      inverted snippets of the components that passed the leakage audit
  per_cell/      per-cell scores and per-claim audit verdicts (numbers and labels)
"""
import os, sys, json, csv, glob, re
os.environ["OPENAI_API_KEY"] = "BLOCKED-cache-only"
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import common as C

OUT = sys.argv[1]
DATASETS = ["askdocs", "askdocs_fever", "biden", "illegal", "onebook", "tifu", "trump"]
TWITTER = ["biden", "illegal", "trump"]
SEEDS = [42, 43, 44]
K, N, TAU = 10, 1000, 0.959
RES = C.EXP_DIR / "results"
for d in ["tweet_ids", "sketches", "snippets", "per_cell"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

def strip_handles(t):
    """Remove @handles the inverter generated; source documents had theirs stripped."""
    return re.sub(r"\s+", " ", re.sub(r"@\w+", "", t)).strip()

def tweet_id(u):
    m = re.fullmatch(r"TWITTER_\d+_(\d+)", u)
    assert m, u
    return m.group(1)

# ---- tweet IDs: whole corpus and each seeded sample, in sample order ----
for ds in TWITTER:
    path, fmt, col = C.DATASETS[ds]
    df = pd.read_csv(path, dtype={"UniversalMessageId": str})
    ids = df["UniversalMessageId"].map(tweet_id)
    open(f"{OUT}/tweet_ids/{ds}_all.txt", "w").write("\n".join(ids) + "\n")
    texts = df[col].dropna().astype(str).map(C.clean_text)
    texts = texts[texts.str.len().between(20, 6000)].drop_duplicates()
    for s in SEEDS:
        samp = texts.sample(N, random_state=s)
        assert samp.tolist() == C.load_corpus(ds, N, s)
        open(f"{OUT}/tweet_ids/{ds}_n{N}_seed{s}.txt", "w").write("\n".join(ids.loc[samp.index]) + "\n")

# ---- sketches and snippets after the nearest-document threshold ----
ref = {(m["dataset"], m["seed"], m["mode"]): m
       for m in json.load(open(RES / "tables" / "stats_medoid_leakage.json"))["leakage_per_mode"]}
rows, kept_total, manifest = [], 0, []
for ds in DATASETS:
    for s in SEEDS:
        corpus = C.load_corpus(ds, N, s)
        emb = np.asarray(C.embed_openai(corpus, cache_tag=f"{ds}_{N}_{s}"), dtype=np.float32)
        gmm = GaussianMixture(K, covariance_type="diag", random_state=s).fit(emb)
        means = gmm.means_.astype(np.float32)
        en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        mn = means / np.linalg.norm(means, axis=1, keepdims=True)
        near = (mn @ en.T).max(axis=1)
        snippets = json.load(open(RES / "e9" / f"{ds}_{N}_{s}_gmm.json"))["conditions"]["vec2summ_gmm"]["texts"]
        for j in range(K):
            assert abs(near[j] - ref[(ds, s, j)]["near_cos"]) < 1e-4, (ds, s, j)
        keep = [j for j in range(K) if near[j] <= TAU]
        kept_total += len(keep)
        np.savez_compressed(f"{OUT}/sketches/{ds}_n{N}_seed{s}.npz",
                            component=np.array(keep), weights=gmm.weights_[keep].astype(np.float32),
                            means=means[keep], diag_covariances=gmm.covariances_[keep].astype(np.float32))
        print("fitted", ds, s, "kept", len(keep), flush=True)
        manifest.append({"dataset": ds, "n": N, "seed": s, "k_fitted": K, "k_released": len(keep)})
        for j in keep:
            r = ref[(ds, s, j)]
            rows.append({"dataset": ds, "n": N, "seed": s, "component": j,
                         "support": round(r["support"], 1), "nearest_document_cosine": round(float(near[j]), 4),
                         "snippet_document_jaccard": round(r["jaccard"], 3), "snippet": strip_handles(snippets[j])})
json.dump(manifest, open(f"{OUT}/sketches/manifest.json", "w"), indent=1)
with open(f"{OUT}/snippets/audited_snippets.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("components kept", kept_total, "of", K * len(DATASETS) * len(SEEDS),
      "| max jaccard among kept", max(r["snippet_document_jaccard"] for r in rows))

# ---- per-cell scores and verdicts, numbers and labels only ----
def wcsv(name, header, data):
    with open(f"{OUT}/per_cell/{name}", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(data)

rec = []
for f in sorted(glob.glob(str(RES / "e1" / "*.json"))) + sorted(glob.glob(str(RES / "e9" / "*.json"))):
    j = json.load(open(f))
    for cond, d in j["conditions"].items():
        if d.get("recon"):
            rec.append([j["dataset"], j["n"], j["seed"], cond, round(d["recon"]["mean_max_sim"], 4)])
wcsv("probe_coverage.csv", ["dataset", "n", "seed", "condition", "mean_max_sim"], rec)

ver = []
for f in sorted(glob.glob(str(RES / "e8" / "*.json"))):
    j = json.load(open(f))
    for cond, a in j["audits"].items():
        for i, c in enumerate(a["claims"]):
            ver.append([j["dataset"], j["n"], j["seed"], cond, i, c["verdict"]])
wcsv("factuality_verdicts.csv", ["dataset", "n", "seed", "condition", "claim_index", "verdict"], ver)

CRIT = ["coverage", "factual_accuracy", "coherence", "conciseness"]
jud = []
for judge, folder, keymap in [("sonnet_gpt41_summaries", "judge_sonnet", "judge_packets"),
                              ("gpt5_llama_summaries", "codex_judge", "codex_packets"),
                              ("sonnet_llama_summaries", "medoid_sonnet", "codex_packets")]:
    km = json.load(open(RES / keymap / "_keymap.json"))
    for f in sorted(glob.glob(str(RES / folder / "*.json"))):
        cell = os.path.basename(f)[:-5]
        ds, s = cell.rsplit("_", 1)
        for label, sc in json.load(open(f)).items():
            jud.append([judge, ds, s, km[cell][label]] + [sc[c] for c in CRIT])
wcsv("cross_judge_scores.csv", ["judge", "dataset", "seed", "condition"] + CRIT, jud)

km = json.load(open(RES / "judge_packets" / "_keymap.json"))
rk = []
for f in sorted(glob.glob(str(RES / "judge_opus" / "*.json"))):
    cell = os.path.basename(f)[:-5]
    ds, s = cell.rsplit("_", 1)
    j = json.load(open(f))
    for crit in ["faithfulness_rank", "coverage_rank"]:
        for pos, label in enumerate(j[crit], 1):
            rk.append([ds, s, crit, pos, km[cell][label]])
wcsv("opus_ranks.csv", ["dataset", "seed", "criterion", "rank", "condition"], rk)
print("rows:", len(rec), len(ver), len(jud), len(rk))
