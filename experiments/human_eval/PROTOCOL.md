# Human-evaluation protocol (Vec2Summ vs. medoid-k)

Anchors the pivotal coverage comparison of Finding 2 with a small source-grounded
study. Design follows SPACE (Angelidis et al., TACL 2021) for the comparative
coverage judgment, SummEval (Fabbri et al., TACL 2021) for the quality
dimensions, and FRANK (Pagnoni et al., NAACL 2021) / Maynez et al. (ACL 2020)
for source-grounded sentence-level faithfulness. No gold reference is required:
the 120-document corpus sample shown is the grounding ("salience pool").

Each item (`items/<cell>.json`) shows a corpus sample (`source_sample`, 120 docs)
and two blinded summaries of that corpus, `summary_S1` and `summary_S2`.
The S1/S2 to system mapping is withheld (`_truth.json`); annotate blind.

## Task 1 Coverage (comparative, the primary anchor)
Read the corpus sample. Decide which summary better reflects the **range of
themes and opinions actually present** in the sample.
- `coverage_winner`: "S1" | "S2" | "tie"
- Also pick a winner (S1/S2/tie) on three secondary criteria (SPACE):
  `coherence` (readable, no contradictions), `conciseness` (useful in few words),
  `non_redundancy` (no repetition).
- `coverage_rationale`: 1-2 sentences naming the themes one summary captured and
  the other missed.

Comparative (forced-choice) is used rather than 1-5 Likert because it is more
reliable at small scale (Kiritchenko & Mohammad, ACL 2017; SummEval found crowd
Likert unreliable).

## Task 2 Faithfulness (source-grounded, per sentence)
For **each sentence** of S1 and of S2 (`S1_sentences`, `S2_sentences`), judge it
against the corpus sample only:
- `supported`: true if the sentence's claim is supported by the sample, false otherwise.
- if false, `error_type`: one of
  - `entity_aspect` wrong entity/aspect (FRANK EntE),
  - `sentiment` wrong polarity/strength of an opinion,
  - `extrinsic` claim absent from the sample (FRANK OutE / Maynez extrinsic),
  - `overgeneralization` overstates prevalence/consensus,
  - `other`.
- Summary `faithfulness_rate` = fraction of its sentences with `supported=true`.

## Output (per item)
```json
{"cell": "...",
 "coverage_winner": "S1|S2|tie",
 "coherence": "...", "conciseness": "...", "non_redundancy": "...",
 "coverage_rationale": "...",
 "S1_faithfulness": {"per_sentence": [{"i":0,"supported":true,"error_type":null}, ...],
                     "faithfulness_rate": 1.0},
 "S2_faithfulness": {"per_sentence": [...], "faithfulness_rate": 0.8}}
```

## Status of the silver pass
`silver/<cell>.json` holds a model-assisted (Claude-Opus) annotation following
this exact protocol. It is **a pilot reference, not human evaluation**: a human
annotator validates/corrects it before any "human-evaluated/validated" claim is
made. Report inter-annotator agreement (Fleiss/Cohen kappa for the binary
faithfulness and coverage judgments) once human labels exist.
