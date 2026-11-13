# Vec2Summ

Code and aggregate results for the paper "Summarizing Without the Documents: Probe Placement, Inversion, and Privacy in Embedding-Sketch Summarization" (EACL 2027, Main Conference), by Mao Li and Johann Gagnon-Bartsch.

Vec2Summ embeds a corpus, fits a k-component Gaussian mixture over the embeddings, inverts each component mean back into text with Vec2Text, and has a language model summarize the k snippets. The paper studies where such a sketch should be probed, what inversion costs against keeping real documents, and what the sketch leaks about individual documents.

## Contents

- `experiments/`: the experiment scripts. `common.py` holds corpus loading, cached embedding and model calls, the mean-max similarity metric and the judge prompts. `e1_run.py` runs the selection baselines and the Gaussian probe scheme, `e9_gmm.py` the component-mean and spectral probes, `e3_gtr.py` the GTR encoder, `e7_hierarchical.py` the hierarchical baseline, `eval_geval.py` the per-criterion judging, `e8_factuality.py` the claim-level audit, `scaling_curve.py` and `scaling_stream.py` the scaling, single-pass and shard-merge results, and the `analysis_*.py` scripts the statistics and the leakage threshold.
- `experiments/results/tables/`: the aggregate tables and statistics behind every table and figure in the paper.
- `experiments/human_eval/`: the expert-panel protocol, the three annotators' recorded judgments and the scoring script's key.
- `figures/`: the scripts that draw the scaling and leakage figures from the tables.
- `requirements.txt`: package versions used for the reported results.

## Data

The corpora are not redistributed here. AskDocs, Reddit-TIFU and the Amazon reviews are available from the sources cited in the paper. Set `VEC2SUMM_DATA` to the folder that holds them; `experiments/common.py` lists the expected file names and text columns. The scripts call the OpenAI API (`OPENAI_API_KEY`) for `text-embedding-ada-002` and GPT-4.1, and an OpenAI-compatible local server for Llama-3.3-70B-Instruct.

This repository contains no source documents, no document-level embeddings and no per-cell files that quote source text. Individual text embeddings are invertible, so document-level embeddings of the Twitter corpora are not released.

## Citation

```
@inproceedings{li-gagnon-bartsch-2027-summarizing,
  title = {Summarizing Without the Documents: Probe Placement, Inversion, and Privacy in Embedding-Sketch Summarization},
  author = {Li, Mao and Gagnon-Bartsch, Johann},
  booktitle = {Proceedings of the 20th Conference of the European Chapter of the Association for Computational Linguistics},
  year = {2027}
}
```

## License

MIT, see `LICENSE`.
