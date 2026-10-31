# Implementation Plan and 15-Day Roadmap

## 1. Target

> **Status note:** this file is the original implementation roadmap, not a live status report. D2, P5, and D1 retrieval-only results are documented in [REPORT.md](REPORT.md); completed items and remaining research work are tracked in [TODO.md](TODO.md).

Deliver a working, reproducible prototype called **Evidence-Aware Adaptive RAG (EA-RAG)** for cardiology educational question answering. The central experiment compares it with fixed top-k RAG on a shared question set.

The baseline paper, *Cardiology-Chat* (2026), uses symptom preprocessing, cardiovascular knowledge retrieval, and a fine-tuned LLM, supported by validated reasoning-data construction. Reproducing its model training, 27k-example dataset, or private 49-case evaluation is outside this plan. The implementation focuses on a narrower, testable system-level hypothesis: evidence-aware retry and abstention may reduce unsupported answers while retaining useful answer coverage.

## 2. Minimum viable contribution

### Research question
Does an evidence-sufficiency gate with one retrieval retry and abstention improve the grounding of cardiology QA responses relative to fixed top-k RAG?

### Hypotheses
- H1: EA-RAG will have a higher evidence-support rate than fixed top-k RAG.
- H2: EA-RAG will produce fewer unsupported answers, at the cost of a higher abstention rate and possibly higher latency.
- H3: Query adaptation/retry will improve retrieval hit rate on questions whose first retrieval is weak.

These are hypotheses, not expected or guaranteed outcomes. Negative results are reportable.

### Method components
1. Fixed top-k RAG baseline.
2. Coarse query intent routing and query normalization/expansion.
3. Evidence sufficiency gate using development-calibrated similarity and support signals.
4. One bounded retry after rewriting/expanding the query.
5. Abstention if evidence remains weak or conflicting.
6. Output citation validator and trace logging.

Avoid overengineering: implement rule-based query routing and simple transparent thresholds first. Do not train a model unless the end-to-end system and evaluation are already working.

## 3. 15-day schedule

The schedule separates an early demonstrable version from follow-up work. Do not misrepresent planned work as already implemented. Reorder if environment constraints demand it.

### Days 1–2: Scope, repo, and baseline reference
**Deliverables**
- Repository scaffold, environment, config, schemas, README.
- Source manifest with 5–10 permitted public cardiology sources.
- Initial question set with explicit provenance.
- A baseline paper summary and initial related-work search notes.

**Acceptance checks**
- Fresh environment installs.
- At least one question can be loaded and validated.
- No private patient data or unlicensed documents.

### Days 3–4: Ingestion and retrieval
**Deliverables**
- Text cleaning, heading-aware chunking, stable IDs, metadata.
- Embedding and local vector index.
- `retrieve(query, k)` interface and simple CLI/debug output.
- Retrieval test cases.

**Acceptance checks**
- Index build is repeatable.
- Each returned chunk includes source title, URL, section, and chunk ID.
- Manually tagged queries can retrieve at least one relevant passage for a pilot subset; report actual performance rather than forcing a threshold.

### Days 5–6: Fixed top-k RAG baseline
**Deliverables**
- Provider adapter for generation.
- Grounded answer prompt.
- Stable source citations.
- Deterministic extractive fallback if no API credentials are configured.
- Baseline output JSONL.

**Acceptance checks**
- End-to-end question → retrieval → answer works.
- Citations reference retrieved chunk IDs.
- Fallback mode is visibly labeled and not mistaken for LLM generation.

### Days 7–8: EA-RAG novelty layer
**Deliverables**
- Intent router (`definition`, `risk_factor`, `symptom_condition`, `test`, `treatment`, `comparison`, `other`).
- Evidence sufficiency scoring with inspectable component signals.
- One retry using query expansion/rewrite.
- Abstention on insufficient/conflicting evidence.
- Citation structural validator.

**Acceptance checks**
- Low-evidence question triggers retry and then abstention when evidence remains insufficient.
- Retry count never exceeds one.
- The same corpus and answer generator are used for baseline and EA-RAG.

### Days 9–10: Evaluation set and metrics
**Deliverables**
- Freeze development/test splits before tuning.
- Evaluation runner that executes both systems against identical questions.
- Metrics: accuracy if answer keys are reliable, retrieval Hit@k/Recall@k where relevance labels exist, citation validity, manually judged evidence support, abstention rate, unsupported-answer rate, latency.
- Per-item JSONL, aggregate CSV/JSON, and experiment configuration.

**Acceptance checks**
- Results are derived from actual runs.
- Every metric states its denominator and definition.
- No test-set threshold or prompt tuning.
- If no reliable answer key exists, do not report answer accuracy.

### Days 11–12: Ablation and error analysis
**Deliverables**
- EA-RAG without abstention.
- EA-RAG without query adaptation/retry.
- Error taxonomy: retrieval miss, stale/conflicting source, unsupported synthesis, invalid citation, unnecessary abstention, missed abstention, parsing failure.
- Small manually reviewed sample with documented rubric.

**Acceptance checks**
- Ablations differ by one component at a time.
- Include examples of successes and failures.
- Avoid treating LLM-as-judge as definitive ground truth.

### Day 13: UI and reproducibility
**Deliverables**
- Streamlit demo showing answer/abstention, sources, and evidence signals.
- Environment/config instructions.
- One command for index build and one command for evaluation.
- Run manifest with timestamp, commit/version, model, corpus version, parameters.

**Acceptance checks**
- Demo starts from a clean setup.
- API keys remain outside the repository.
- Research-only notice appears in the interface.

### Day 14: Paper-ready artifacts
**Deliverables**
- Tables/plots from actual outputs.
- Method and system diagram.
- Results, limitations, and error analysis draft.
- Related-work notes with verified citations.
- Final test run.

**Acceptance checks**
- No unrun experiment is represented as a result.
- Every numerical claim traces to an artifact.
- Compare against the baseline paper carefully; do not claim a fair head-to-head comparison unless the exact benchmark and protocol are replicated.

### Day 15: Freeze and handoff
**Deliverables**
- Tagged code snapshot or release.
- Final README, reproducibility notes, data manifest, and experiment outputs.
- Short demo script and a list of remaining limitations.
- Paper draft with contributions stated conservatively.

**Acceptance checks**
- A clean checkout reproduces the reported pipeline.
- All figures and tables have source data.
- Limitations and future work are explicit.

## 4. What to implement first (vertical slice)

Before building everything, make this work:
1. Load 3–5 local, permitted documents.
2. Chunk and index them.
3. Ask one test question.
4. Retrieve top passages.
5. Generate an answer with citations (or run labeled extractive fallback).
6. Run the evidence gate and show abstention for a deliberately out-of-corpus question.
7. Save a JSON record of the run.

Only then expand the corpus and evaluation set.

## 5. Suggested initial dataset schema

`data/evaluation/dev.jsonl` and `test.jsonl`, one JSON object per line:

```json
{
  "question_id": "q001",
  "question": "What is ...?",
  "category": "definition",
  "reference_answer": "Short reference answer or null",
  "relevant_chunk_ids": [],
  "source_url": "https://...",
  "split": "dev"
}
```

Do not leave `relevant_chunk_ids` empty if reporting retrieval metrics for that item. Do not create a reference answer by asking the evaluated model and then treat it as ground truth. Maintain source notes and provenance.

## 6. Experiment protocol

- Freeze corpus version, dev/test split, model/provider, prompt version, and parameters.
- Tune thresholds on dev only.
- Run fixed top-k RAG and EA-RAG on identical test questions.
- Keep model temperature and decoding settings fixed.
- Store raw response, retrieved chunk IDs, similarity values, retry event, abstention reason, citations, latency, and run config.
- Compute metrics from saved raw outputs with a separate script.
- If API responses are nondeterministic, report the model/version, settings, date, and any repeated-run protocol.
- Use paired comparisons where appropriate; include uncertainty rather than overclaiming from a small test set.

## 7. Paper outline

1. **Title:** Evidence-Aware Adaptive Retrieval and Abstention for Grounded Cardiology Question Answering
2. **Abstract:** Problem, method, dataset, actual results, limitations.
3. **Introduction:** Medical-domain grounding challenge; gap and research question.
4. **Related Work:** Cardiology-Chat, medical RAG, adaptive retrieval, abstention/selective prediction, citation verification.
5. **Method:** Baseline top-k RAG, query routing, evidence gate, retry, abstention, citation validation.
6. **Experimental Setup:** Sources, questions, split policy, models, prompts, metrics, hardware/API setup.
7. **Results:** Main comparison, ablations, retrieval metrics, support/abstention trade-off, latency.
8. **Error Analysis:** Representative grounded, unsupported, conflicting-evidence, and abstention examples.
9. **Limitations and Safety:** Small corpus, small benchmark, source coverage, model dependence, no clinical validation.
10. **Conclusion and Future Work.**

## 8. Suggested result tables

### Main results
| System | Answer accuracy (if valid key) | Evidence support rate | Unsupported-answer rate | Abstention rate | Citation validity | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| Fixed top-k RAG | measured | measured | measured | measured | measured | measured |
| EA-RAG | measured | measured | measured | measured | measured | measured |

### Ablation
| Variant | Evidence support rate | Unsupported-answer rate | Abstention rate | Hit@k | p50 latency |
|---|---:|---:|---:|---:|---:|
| EA-RAG full | measured | measured | measured | measured | measured |
| Without abstention | measured | measured | measured | measured | measured |
| Without adaptation/retry | measured | measured | measured | measured | measured |

Fill cells only with measured results. Include denominators and definitions in the paper.

## 9. Risks and fallback plans

- **No API key:** use extractive fallback to validate retrieval, citations, abstention, and evaluation plumbing. Do not claim this equals generative RAG.
- **Embedding download blocked:** use a local embedding model if available; otherwise implement a BM25/TF-IDF baseline and clearly label the substitution.
- **Insufficient corpus:** keep scope small, manually validate the corpus, and emphasize system evaluation rather than clinical conclusions.
- **Too little time:** prioritize fixed RAG + evidence gate + one retry + abstention + paired evaluation. Leave UI polish and advanced reranking for later.
- **Unclear novelty:** search recent literature on adaptive RAG, selective prediction/abstention, evidence-grounded QA, and citation verification; refine the contribution to a narrow measurable gap.
- **Weak results:** report them honestly and analyze why. A useful negative result is preferable to unsupported claims.

## 10. Work intentionally left for later milestones

These are future tasks, not completed features:
- Broader source coverage and automated update checks.
- More robust contradiction detection and reranking.
- Expert review of evidence-support annotations.
- Larger question sets and external validation.
- Multilingual queries and multimodal ECG/echo inputs.
- Fine-tuning or specialized model training.
- More formal statistical analysis and calibration.
