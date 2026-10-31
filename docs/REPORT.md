# EA-RAG Cardiology Evaluation: Experiment Report

**Status:** completed exploratory evaluation; not clinical validation  
**Evaluation snapshot:** 1 November 2026  
**Primary benchmark:** D2 test, 609 MCQs  
**Generator:** `gemma4:31b` through Ollama Cloud  
**Retriever used in reported runs:** local lexical BM25-style retrieval  
**Primary knowledge base:** D4 hierarchical, 1,055 pre-chunked passages

This document records the methods, implementation, data inventory, observed results, and limits of the current experiment. It is intended as a reproducibility and paper-planning reference. The numbers below come from the saved JSONL traces and metrics in `data/artifacts/`; they should not be presented as evidence of clinical safety or diagnostic capability.

## 1. Executive summary

This project evaluates a small evidence-aware retrieval policy for cardiology-oriented question answering. The policy uses a fixed top-4 retriever, an evidence-score gate, at most one intent-based query expansion and retry, then abstains if the best available passage still falls below a threshold. The comparator is fixed top-4 retrieval without the evidence gate.

The final D2 test comparison includes 609 questions and uses Gemma 4 31B for generation. The fixed baseline answered all items and scored **555/609 (91.1%)**. EA-RAG abstained on **78/609 (12.8%)**, answered 531, and scored **487/609 (80.0%)** when abstentions count as incorrect. On the questions it did answer, EA-RAG scored **487/531 (91.7%)**. The baseline scored **485/531 (91.3%)** on that same answered subset; the paired difference was not statistically significant (exact McNemar *p* = 0.6875).

This is not evidence that EA-RAG is more accurate than fixed RAG. Its selective accuracy was only 0.6 percentage points higher, while coverage fell by 12.8 points. For 74 of the 78 abstentions, a saved no-abstention candidate answer was correct. The score threshold therefore rejected many answers that the model would have answered correctly. The no-abstention ablation scored 561/609 (92.1%), a 1.0-point increase over fixed baseline that was not significant in the paired exact test (*p* = 0.1094).

On the 21-item P5 anchor-stripping subset, fixed and EA-RAG both scored 19/21 (90.5%); neither retried nor abstained. The current P5 result does not support the hypothesis that anchor stripping raises EA-RAG's abstention rate. A local D1 retrieval-only audit yielded 90% post-retry coverage, but diagnostic accuracy was not measured because no validated diagnosis matcher is implemented.

**Research implication:** the code and evaluation harness now run end to end, and the selective-prediction question can be studied. The current threshold policy has not shown a meaningful answer-quality advantage. The defensible contribution is an application-specific empirical evaluation and its coverage/robustness findings—not a new retrieval or abstention algorithm.

## 2. Research question, baseline paper, and contribution

### 2.1 Reference paper

The project takes inspiration from **Cardiology-Chat: A Multi-LLMs Powered System for Cardiac Diagnostic Reasoning and Clinical Support** (Yang et al., 2026; DOI [10.1109/JTEHM.2026.3668755](https://doi.org/10.1109/JTEHM.2026.3668755); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41970942/); [full-text landing page](https://pmc.ncbi.nlm.nih.gov/articles/PMC13068123/)). That paper describes a cardiology-focused, multi-stage system with structured knowledge retrieval and model-generated reasoning. The repository's prior paper summary lists its evaluation as 182 cardiology MCQs and 49 anonymized in-house cases; the paper reports 0.796 average QA accuracy and 0.807 case F1 for its structured-KB system. Its data, training, model pipeline, and experiments are not reproduced here. Those headline scores are not directly comparable with this repository because the model, benchmark, corpus, prompts, and protocol differ.

The term **fixed baseline** in this report means this repository's own fixed top-k RAG arm. It does not mean a reproduction of Cardiology-Chat.

### 2.2 Research question

> On a fixed cardiology-oriented guideline corpus and MCQ set, does a lexical-score evidence gate with one bounded query retry and abstention improve selective answer quality over fixed top-k RAG, and how does it behave under a small anchor-stripping perturbation?

### 2.3 Novelty positioning

Retrieval gating, iterative/corrective retrieval, and abstention are established research directions. Self-RAG studies retrieval-on-demand and self-reflection; CRAG assesses retrieval quality and triggers corrective actions; Adaptive-RAG changes retrieval strategy according to query complexity. Selective prediction is also commonly described as a risk/coverage trade-off. These works rule out a broad claim that “abstain instead of guessing” or adaptive retrieval is new ([Self-RAG](https://arxiv.org/abs/2310.11511), [CRAG](https://arxiv.org/abs/2401.15884), [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/), [selective classification](https://papers.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf)).

The defensible contribution is narrower:

1. An auditable, cardiology-oriented evaluation setup combining public MCQ sources, a small fixed guideline chunk corpus, and a targeted clinical-anchor perturbation.
2. A paired comparison of fixed RAG, one-retry EA-RAG, and two ablations, reporting answer accuracy together with coverage and abstention.
3. An empirical result showing that this lexical score threshold does **not** reliably identify wrong answers: on D2 test, 74/78 saved counterfactual answers for abstained items were correct.
4. A negative robustness finding: the P5 subset did not trigger the gate and showed no accuracy difference between fixed RAG and EA-RAG.

This is a **candidate system-evaluation contribution**, not proof of a new mechanism or a validated new benchmark. Claims should remain provisional until the dataset provenance, D1 labels, source permissions, and threshold behavior receive further review.

## 3. Prepared dataset inventory and audit

The local `data/` directory contains the frozen artifacts below. SHA-256 values are supplied to make the inputs to the reported runs identifiable. The full dataset audit is in `data/audit_report.txt`; the data README and manifests contain additional schemas and details.

| Asset | Size and contents | Role in this experiment | Important qualification |
|---|---|---|---|
| D1 staged cases | 100 case reports; 16 from 2025 and 84 from 2026 | Local Stage 1 retrieval/coverage audit only | Not diagnostic-accuracy scored; title/abstract overlap is high and no validated matcher is present |
| D2 train | 4,910 MCQs | Not used for inference | Do not use as a test set |
| D2 validation | 613 MCQs: 273 MedQA, 340 MedMCQA | 100-item retrieval-score calibration sample; separate 20-item threshold pilot | Calibration was based on score quantiles, not correctness labels |
| D2 test | 609 MCQs: 273 MedQA, 336 MedMCQA | Primary four-arm evaluation | 555 baseline answers correct; answer distribution is broad academic QA, not a pure cardiology-only clinical cohort |
| D3 P5 | 21 anchor-stripped MCQs | Fixed vs. EA-RAG robustness check | Small subset; threshold did not trigger on any item |
| D4 fixed chunks | 959 passages | Not used for headline experiment | IDs match D5 candidate pairs |
| D4 hierarchical chunks | 1,055 passages with stable IDs/headings | Retrieval corpus for all reported LLM runs | Pre-chunked guideline-derived passages; this experiment did not re-ingest or re-split the PDFs |
| D5 retrieval pairs | 46 queries with candidate positive IDs | Not evaluated in this experiment | IDs match fixed D4, not hierarchical D4; pairs were generated heuristically, not human-judged gold relevance labels |
| D6 progressive reveal | 100 trajectories | Not evaluated | 100/100 link to D1 by DOI and are cumulative; they are not an independent case cohort |

### 3.1 D2 split and quality notes

The D2 files have valid four-option records and answer letters. The audit reports no exact duplicate questions inside the splits, no exact question overlap between train/validation/test, and no near-duplicate overlap at its word-Jaccard threshold of 0.80. The split builder sorts questions by text and then takes deterministic contiguous slices within source; this is reproducible, but it is not random or stratified sampling.

The D2 test contains 273 MedQA items and 336 MedMCQA items. The audit found a cardiology term in 269/273 MedQA stems and 322/336 MedMCQA stems. It also flags four OCR-damage markers in MedMCQA test and four in validation, despite the “clean” filename. The dataset should therefore be described as a cardiology-oriented mixed academic MCQ set, not as 609 expert-authored cardiology cases.

### 3.2 D1 and D6 limitations

D1 has 100 CC-BY-labeled case reports, but its staged summaries are short and sometimes overlap with the source abstract: the audit finds an 8-gram shared with the abstract in **91/100 Stage 1** records and mean title-word overlap of 0.31. A crude lexical coverage check finds all key words of only **2/97** usable D1 gold diagnoses inside a single hierarchical D4 chunk (1/97 in fixed D4). This suggests D1 is a useful out-of-corpus stress set, but not yet a supported diagnosis-accuracy benchmark for this KB.

D6 is derived from the same D1 cases (100/100 DOI-linked; mean word-Jaccard similarity to D1 stages 0.82). It must not be described as independent validation.

### 3.3 D4 provenance and use

The hierarchical corpus has **1,055 stable chunks** from three local guideline documents: ESC acute coronary syndromes (2023), ESC heart failure (2026), and ACC/AHA valvular heart disease (2020). The loader preserves `chunk_id`, `section_heading`, and `source_doc` using `data/manifests/D4_sources.json`.

The manifest records restrictions on reuse and reproduction. Before submitting or distributing derived text, figures containing quotations, or artifacts with passage text, verify the applicable ESC/AHA permissions and the model provider's data-handling terms. The D4 source PDFs and chunk text should remain local unless the relevant terms permit external processing and redistribution.

### 3.4 Frozen file hashes

| File | Rows | SHA-256 |
|---|---:|---|
| `D2_train_clean.jsonl` | 4,910 | `e0f3ffe98b28a36d4c76c1fc581360da689d19fe645d4537ebbf89dd59abc154` |
| `D2_val_clean.jsonl` | 613 | `c306b38e2f25ba7144d18fa1c7a055ab4a75f0ad96b611acab7a1f8ff06be4d1` |
| `D2_test_clean.jsonl` | 609 | `e2fa56d15a5bf85e9e22e3de96ce554586a86165e2b8ae48823a67603e6437b1` |
| `D1_cases_staged.jsonl` | 100 | `d92e30b91a1f36a7724444612dae79676d8a481cfd4a6854027cfd06f32ee66c` |
| `D3_P5_stripped.jsonl` | 21 | `ef4cba0fe48d0cd6ce420d57f3849a63bab6be8d19eeb45ef7c2ebc126daa20d` |
| `D4_cardiology_kb_hierarchical.jsonl` | 1,055 | `949364fad322bc5aa436733d84f5e24f8c76149c6a78c32d3d9ca3257b413b3b` |

## 4. System and experimental method

### 4.1 Data-to-answer path

1. Read the frozen D4 hierarchical JSONL directly; preserve its chunk IDs and attach source metadata from the D4 manifest.
2. Build a deterministic local lexical index over chunk text.
3. For each MCQ, retrieve top 4 using the **question stem only**. The options are sent to the generator but are not appended to the retrieval query (`include_options_in_query=false`).
4. Pass retrieved passages, the stem, and A–D options to `gemma4:31b` using Ollama Cloud. The MCQ prompt requests exactly `FINAL ANSWER: A` (or B/C/D) at temperature 0.
5. Parse the final answer with a strict letter parser. In MCQ mode, citations are optional; if the model includes chunk IDs, they are checked against the retrieved IDs. A missing citation no longer forces an abstention. The earlier pilot exposed this mismatch and the MCQ path was corrected before the final runs.
6. Compare the parsed letter with the D2 gold answer. An abstention counts as incorrect for ordinary accuracy and is also reported separately through coverage and selective accuracy.

### 4.2 What retrieval was actually used

The reported experiments used `LexicalIndex`, a deterministic **BM25-style lexical ranker**, not embeddings or a vector database. For each chunk, the ranker computes a BM25-like term score and maps the positive score to `score / (score + 2)`. The resulting value lies in `[0,1)` and is a ranking heuristic, not a calibrated probability of support or correctness. `DenseIndex` and sentence-transformer cosine retrieval exist as optional code paths, but no dense model or vector index was used for the results in this report. There was no reranker.

The default retrieval cutoff in the frozen run was `top_k=4`. The evaluation corpus was the hierarchical D4 file, not the 959-chunk fixed corpus. D5 was not used to tune the result because its candidate chunk IDs do not match hierarchical D4 and its labels have not been human-validated.

### 4.3 Evidence thresholds and retry

To avoid generation calls during initial calibration, 100 records were sampled from D2 validation using seed 17. The top-1 lexical score distribution was:

| Statistic | Score |
|---|---:|
| Minimum | 0.8745 |
| 10th percentile | 0.9256 |
| 15th percentile | 0.9294 |
| Median | 0.9566 |
| 85th percentile | 0.9676 |
| Maximum | 0.9792 |

The frozen values were `min_score=0.9256` for final evidence sufficiency and `retry_min_score=0.9294` for triggering the one retry. If the initial top score is below the retry cutoff, the pipeline expands the query using its coarse intent router, retrieves once more, merges/deduplicates results, and then applies the lower abstention threshold. The no-adaptation ablation keeps the initial retrieval and evidence gate but skips this retry. The no-abstention ablation retains retry but returns a candidate answer even when the final gate would abstain.

This procedure sets thresholds by **score percentiles only**. It does not establish that the bottom 10–15% of scores correspond to wrong answers. The D2 test results show that this assumption is weak for this ranker and corpus.

### 4.4 Two-pass execution and derived ablations

The fixed baseline was run once per question (609 calls). EA-RAG was then run once per question. Its trace stores the post-retry candidate used for the no-abstention ablation on the 78 items where the production EA-RAG path abstained. For the no-adaptation ablation, the trace additionally stores an initial-retrieval answer for the 17 cases where retry was triggered despite the initial evidence gate passing. This required **17 additional counterfactual model calls** beyond one EA-RAG call per item. Total D2 primary/counterfactual calls were therefore 609 + 609 + 17 = **1,235**, rather than 2,436 calls for four independent arms.

The derived arms are paired by question ID using `scripts/derive_two_pass_ablation.py`. An abstention counts as incorrect for overall accuracy. Selective accuracy uses only answered rows. Paired exact McNemar tests use item-level correctness; they do not assess semantic evidence support.

### 4.5 Generator and run configuration

| Setting | Value |
|---|---|
| Generator | `gemma4:31b` via Ollama Cloud OpenAI-compatible endpoint |
| Temperature | 0 |
| Request timeout | 45 s default |
| Retrieval backend | lexical BM25-style |
| Corpus | D4 hierarchical, 1,055 chunks |
| Top-k | 4 |
| MCQ retrieval query | stem only |
| Threshold tuning | 100 D2 validation records, seed 17; score percentiles |
| Test size | 609 |
| Generator errors | 0 on full D2 baseline and EA-RAG runs |

The run manifests identify corpus and source-manifest hashes, model name, retrieval method, thresholds, and question count. The cloud model's exact serving revision was not captured, which limits perfect inference reproducibility. The prompts contained the MCQ question/options and retrieved context; D4-derived passages were therefore sent to Ollama Cloud during these experiments. Confirm source-use and provider data-processing terms before repeating the run or publishing/reusing passage content.

## 5. Results

### 5.1 Main D2 test comparison

Overall accuracy uses all 609 questions and counts abstentions as incorrect. Selective accuracy uses only non-abstained answers. Latency is the mean primary-path latency from run records; counterfactual overhead is excluded from this per-answer statistic.

| Arm | Correct / N | Overall accuracy (95% Wilson CI) | Coverage | Abstention rate | Selective accuracy | Retry rate | Mean primary latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed top-4 baseline | 555/609 | 91.1% (88.6–93.1%) | 100.0% | 0.0% | 91.1% | 0.0% | 1.843 s |
| EA-RAG | 487/609 | 80.0% (76.6–83.0%) | 87.2% (531/609) | 12.8% (78/609) | 91.7% (487/531; 89.1–93.8%) | 19.4% (118/609) | 1.621 s |
| EA-RAG without abstention | 561/609 | 92.1% (89.7–94.0%) | 100.0% | 0.0% | 92.1% | 19.4% | 1.816 s |
| EA-RAG without adaptation/retry | 465/609 | 76.4% (72.8–79.6%) | 83.4% (508/609) | 16.6% (101/609) | 91.5% (465/508; 88.8–93.7%) | 0.0% | 1.623 s |

P50 primary latencies were 1.830 s (fixed), 1.633 s (EA-RAG), 1.748 s (no abstention), and 1.633 s (no adaptation). Total wall time for the two full passes was approximately 18.7 minutes for fixed and 18.9 minutes for EA-RAG, including retry/counterfactual work. Both 609-record passes had zero run errors; the maximum complete record latency was 8.68 s for fixed and 5.96 s for EA-RAG.

### 5.2 Paired comparisons and interpretation

| Paired comparison | Discordant items | Exact McNemar *p* | Interpretation |
|---|---:|---:|---|
| Fixed vs. EA-RAG, all items (abstentions wrong) | 72 fixed-only correct; 4 EA-only correct | 3.59 × 10⁻¹⁷ | Overall correctness favors fixed because EA abstains on some items the fixed arm answers correctly |
| EA-RAG vs. no-abstention | 0 EA-only correct; 74 no-abstention-only correct | 1.06 × 10⁻²² | The withheld candidate answers are mostly correct on this benchmark; this test does not show that abstention improves overall accuracy |
| EA-RAG vs. no-adaptation | 22 EA-only correct; 0 no-adaptation-only correct | 4.77 × 10⁻⁷ | Retry/adaptation recovers some items and improves the all-item correctness outcome over the no-adaptation ablation |
| Fixed vs. EA-RAG on EA-answered subset (n=531) | 2 fixed-only correct; 4 EA-only correct | 0.6875 | No detectable answer-quality difference among the items EA-RAG chose to answer |
| Fixed vs. no-abstention | 2 fixed-only correct; 8 no-abstention-only correct | 0.1094 | The 1.0-point difference is not significant at the conventional 0.05 level |

The prominent result is the coverage/accuracy trade-off: EA-RAG reduces coverage to 87.2%, yet selective accuracy increases only from 91.1% to 91.7%. This difference is not significant on the shared answered subset. In addition, 74/78 (94.9%) of EA-RAG's saved candidate answers for abstained items were correct. The current lexical score gate therefore does not discriminate errors well enough to justify a safety or reliability claim.

The no-adaptation ablation has lower coverage (83.4%) than full EA-RAG (87.2%). Its selective accuracy (91.5%) is close to EA-RAG's (91.7%). This suggests the retry recovered coverage more clearly than it improved accuracy among answered questions in this run.

### 5.3 D3 P5 anchor-stripping stress test

| Arm | Correct / N | Accuracy (95% Wilson CI) | Coverage | Abstentions | Retries |
|---|---:|---:|---:|---:|---:|
| Fixed baseline | 19/21 | 90.5% (71.1–97.4%) | 100% | 0 | 0 |
| EA-RAG | 19/21 | 90.5% (71.1–97.4%) | 100% | 0 | 0 |

Relative to each item's unperturbed D2 test prediction, the fixed arm changed answer on 2/21 items (9.5%) and EA-RAG on 1/21 (4.8%). The P5 questions did not fall below either threshold; no item retried or abstained. The subset is small and provides no evidence that this EA-RAG configuration detects the removed anchors. It also does not demonstrate that the model is robust to anchor removal; accuracy is unchanged here, but only 21 cases were tested.

### 5.4 D1 Stage 1 retrieval-only audit

| Measure | Result |
|---|---:|
| Stage 1 cases | 100 |
| Initial retrievals below retry cutoff | 23 (23%) |
| Final abstentions after at most one retry | 10 (10%) |
| Post-retry coverage | 90% |
| Median initial top-1 score | 0.9464 |
| Diagnostic accuracy | Not measured |

This was a local retrieval/evidence-gate simulation only; no LLM diagnosis generation was run on D1. The project has no validated diagnosis matcher, and `gold_primary` is derived from case-report titles/text rather than a separately adjudicated structured diagnosis ontology. The D1 audit also flags 8-gram overlap between Stage 1 and abstract in 91/100 cases. Do not report D1 diagnostic accuracy until a matcher, leakage analysis, and evaluation protocol are validated.

### 5.5 Metrics not measured

- No dense embedding or vector database result.
- No D5 Recall@k/Hit@k result; D5 candidates are heuristic and IDs do not align with hierarchical D4.
- No semantic citation-entailment or evidence-support human review. MCQ citation checks only reject explicit IDs that are not in the retrieved set; citations are optional in MCQ mode. Reported “citation validity” values must not be interpreted as groundedness.
- No clinician-judged diagnosis accuracy on D1, no D6 trajectory score, and no evidence that model outputs are safe for patient care.
- No full risk–coverage curve. The D2 table reports the frozen operating point and the paired answered-subset comparison.

## 6. Dataset and pipeline changes completed

The evaluation work required several adapters and fixes:

1. **Pre-chunked D4 loading:** `read_chunks()` accepts the existing D4 JSONL schema, keeps stable chunk IDs, and joins source provenance from `D4_sources.json`. This avoids rerunning PDF splitting.
2. **D2 MCQ support:** the evaluator validates A–D options and gold letters, sends options to generation, parses a strict `FINAL ANSWER: [A-D]` line, and computes correctness/coverage.
3. **MCQ citation mismatch fix:** a missing citation no longer suppresses a clean MCQ letter. If a response includes citations, they must refer to retrieved chunks. Open-ended QA citation checks remain structural and are not proof of semantic support.
4. **Separate retry and abstention thresholds:** retry threshold is above final abstention threshold; the manifest records both.
5. **Single-arm evaluation and checkpointed JSONL:** the CLI supports `--variant fixed` or `--variant ea_rag`, and writes each completed item so a long run retains partial traces if interrupted.
6. **Counterfactual trace and ablation script:** EA-RAG logs preserve the candidates needed to derive no-abstention and no-adaptation arms; `scripts/derive_two_pass_ablation.py` creates the four-arm comparison and exact paired McNemar tests.
7. **D1 local score audit:** `scripts/score_d1_stage1_retrieval.py` records scores, retries, final gate decisions, and case IDs without claiming diagnostic accuracy.
8. **Unicode display check:** the D2 validation and hierarchical D4 files contain no actual U+FFFD replacement characters. PowerShell output was set to UTF-8 for subsequent runs; no source corpus rewrite was made for the earlier console-rendered `�` symbol.

The included test suite passes **15/15** tests in the uv environment. This validates code behavior covered by those tests; it is not clinical validation.

## 7. Reproduction commands

Run commands from the repository root in PowerShell. They use the project virtual environment through `uv`; Ollama Cloud credentials are read from `.env` or process environment and are not included in artifacts.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:PYTHONPATH = 'src'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Retrieval-only threshold calibration was saved as:
# data/artifacts/retrieval_calibration_d2_val100.json

# Fixed pass
uv run --python .venv\Scripts\python.exe -m cardio_rag.cli evaluate `
  --questions data\artifacts\D2_test_clean.jsonl `
  --run-id d2_test_fixed --variant fixed --retrieval lexical `
  --corpus 'data\knowledge base\indexed chunks\D4_cardiology_kb_hierarchical.jsonl' `
  --source-manifest data\manifests\D4_sources.json --output data\artifacts

# EA-RAG pass
uv run --python .venv\Scripts\python.exe -m cardio_rag.cli evaluate `
  --questions data\artifacts\D2_test_clean.jsonl `
  --run-id d2_test_earag --variant ea_rag --retrieval lexical `
  --corpus 'data\knowledge base\indexed chunks\D4_cardiology_kb_hierarchical.jsonl' `
  --source-manifest data\manifests\D4_sources.json --output data\artifacts

# Derive the four-arm table and paired tests
uv run --python .venv\Scripts\python.exe scripts\derive_two_pass_ablation.py `
  --baseline data\artifacts\d2_test_fixed.jsonl `
  --ea-rag data\artifacts\d2_test_earag.jsonl `
  --output data\artifacts\d2_test_two_pass_comparison.jsonl
```

Core outputs:

- `data/artifacts/d2_test_fixed.jsonl`, `_metrics.json`, `_manifest.json`
- `data/artifacts/d2_test_earag.jsonl`, `_metrics.json`, `_manifest.json`
- `data/artifacts/d2_test_two_pass_comparison.jsonl` and `_metrics.json`
- `data/artifacts/d3_p5_comparison_metrics.json`
- `data/artifacts/d1_stage1_retrieval_audit.json`
- `data/artifacts/retrieval_calibration_d2_val100.json`

## 8. Paper framing and six-page plan

### Suggested title

**Score-Gated Retrieval, Retry, and Abstention for Cardiology-Oriented Question Answering: A Selective-Prediction Evaluation**

Avoid a title that claims clinical diagnosis, clinical safety, a new general RAG algorithm, or superiority over Cardiology-Chat.

### Suggested six-page structure

1. **Introduction (about 0.6 page):** cardiology knowledge coverage and unsupported answers; research question; state explicitly that selective RAG is established and this work studies a narrow cardiology-oriented setup.
2. **Related work (about 0.7 page):** Cardiology-Chat; Self-RAG, CRAG, Adaptive-RAG; selective prediction and risk–coverage evaluation. Explain how this system differs in scope and simplicity, not in claiming a new mechanism.
3. **Methods (about 1.2 pages):** frozen D4 corpus, lexical scoring, thresholds, query expansion, one retry, MCQ prompt/parser, abstention, baseline and ablations.
4. **Data and experimental setup (about 0.9 page):** D1–D6 inventory; D2 split sources; model/provider; prompt and decoding; hashes; metrics and paired tests. Put full artifact hashes in supplementary material if space is tight.
5. **Results (about 1.1 pages):** main four-arm D2 table; coverage/selective accuracy; confidence intervals and paired McNemar tests; P5 and D1 retrieval-only findings.
6. **Discussion, limitations, and conclusion (about 1.5 pages):** threshold failure signal (74/78 counterfactual answers correct); no significant selective-accuracy lift; no P5 abstention; dataset/retriever/model limitations; careful future work.

### Suggested PPT sequence (8 slides)

1. Motivation and exact research question.
2. Cardiology-Chat reference and related work; what this project does and does not claim.
3. Dataset map: D2/D3/D4 primary; D1/D5/D6 exploratory or not used.
4. Pipeline diagram: top-4 lexical retrieval → score gate → one rewrite/retry → answer or abstain.
5. Evaluation protocol: Gemma 4 31B, threshold calibration, two-pass logs, ablations.
6. D2 main results table and coverage–accuracy interpretation.
7. P5 and D1 findings; emphasize the negative/limited results.
8. Limitations, defensible contribution, and next experiments.

## 9. Limitations and next research steps

1. **Revisit threshold selection using D2 validation answer outcomes.** Percentile thresholds do not indicate correctness; the 74/78 counterfactual result is a warning. Do all threshold selection on D2 validation and preserve D2 test as a one-time fixed evaluation. The current D2 test has already been inspected and must not be used to retune and then presented as an untouched holdout.
2. **Build a labeled retrieval benchmark for the hierarchical corpus.** Human-review D5 query/passage relevance labels or construct a new set whose positive IDs match hierarchical D4.
3. **Generate a risk–coverage curve on validation data.** Report risk at fixed coverage and coverage at a fixed target risk; select an operating point before test evaluation.
4. **Obtain human evidence-support annotations.** Structural chunk-ID validity is not entailment. Review a blinded sample with a written rubric and at least two reviewers if feasible.
5. **Validate D1 diagnosis evaluation.** Define a normalized diagnosis ontology/matcher, inspect abstract/title leakage, and establish adjudicated references before computing case accuracy. Keep retrieval-only results separate.
6. **Expand robustness testing.** P5 has only 21 questions and caused no gate events. Run other perturbations, report paired answer-flip rates, and analyze whether perturbations preserve clinical meaning.
7. **Evaluate dense retrieval separately.** If permitted, install/cache a pinned embedding model, record its exact revision, and compare lexical and dense retrieval on the same manually labeled relevance set. Do not call the lexical scores cosine similarities.
8. **Record model service version and cost.** Ollama Cloud returned model name `gemma4:31b`, but no immutable serving revision or token/cost record is available in this run.
9. **Audit data rights and privacy.** D4 source terms restrict reproduction and derived-content reuse. Confirm that external API processing of retrieved passages is permitted before repeating or sharing those runs.

## 10. References

1. Yang Z, Chen C, Mahmoud SS, Tan X, Chen Y, Fang Q. *Cardiology-Chat: A Multi-LLMs Powered System for Cardiac Diagnostic Reasoning and Clinical Support*. IEEE Journal of Translational Engineering in Health and Medicine. 2026;14:123–132. [DOI: 10.1109/JTEHM.2026.3668755](https://doi.org/10.1109/JTEHM.2026.3668755).
2. Asai A, Wu Z, Wang Y, Sil A, Hajishirzi H. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511).
3. Yan S-Q, Gu J-C, Zhu Y, Ling Z-H. *Corrective Retrieval Augmented Generation*. 2024. [arXiv:2401.15884](https://arxiv.org/abs/2401.15884).
4. Jeong S, Baek J, Cho S, Hwang SJ, Park J. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. NAACL-HLT 2024:7036–7050. [DOI: 10.18653/v1/2024.naacl-long.389](https://doi.org/10.18653/v1/2024.naacl-long.389).
5. Geifman Y, El-Yaniv R. *Selective Classification for Deep Neural Networks*. NeurIPS 2017. [Proceedings paper](https://papers.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf).

