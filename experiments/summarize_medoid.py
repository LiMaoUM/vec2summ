"""Generate medoid-k summaries via local vLLM (Llama-3.3-70B). No OpenAI calls.
medoid-k = the real corpus document nearest each GMM mode (texts from
tables/medoid_texts.json). Same summarizer + prompt + length as every other
condition, so the result slots into the Llama-summarizer comparison.
"""
import os, json
os.environ["OPENAI_API_KEY"] = "dummy"  # only to satisfy client import; we call local only
import common as C

RES = C.EXP_DIR / "results"
medoid_texts = json.load(open(RES / "tables" / "medoid_texts.json"))
OUT = RES / "e10_medoid"
OUT.mkdir(exist_ok=True)

done = 0
for cell, texts in medoid_texts.items():
    f = OUT / f"{cell}.json"
    if f.exists():
        done += 1
        continue
    summ = C.summarize(texts, words=150, local=True)
    json.dump({"cell": cell, "k": len(texts), "summary_llama": summ,
               "medoid_texts": texts}, open(f, "w"))
    done += 1
    print(f"[{done}/{len(medoid_texts)}] {cell}: {(summ or '')[:70]}...", flush=True)
print(f"medoid summaries written: {done}")
