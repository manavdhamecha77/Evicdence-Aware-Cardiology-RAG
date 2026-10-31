# Evidence-Aware Cardiology QA: System Architecture

## 1. Purpose

This repository contains a reproducible research prototype inspired by *Cardiology-Chat: A Multi-LLMs Powered System for Cardiac Diagnostic Reasoning and Clinical Support* (IEEE Journal of Translational Engineering in Health and Medicine, 2026, DOI: 10.1109/JTEHM.2026.3668755). The completed evaluation and measured limitations are documented in [REPORT.md](REPORT.md).

The baseline paper combines (1) symptom/query preprocessing, (2) retrieval from a cardiovascular vector knowledge base, and (3) a fine-tuned LLM for a final response. It also describes generator–validator reasoning-data construction. Its reported evaluation uses 182 cardiology multiple-choice questions and 49 anonymized clinical cases; reported headline results are 0.796 QA accuracy and 0.807 clinical-case F1. These are paper-reported results, not targets or results of this implementation.

This project implements a smaller, auditable prototype; it does not reproduce the original model training or private clinical evaluation.

## 2. Research framing and novelty limits

**Primary research question:** On a fixed cardiology-oriented guideline corpus and MCQ set, does a lexical-score evidence gate with one bounded retry and abstention improve selective answer quality over fixed top-k RAG?

The implemented **Evidence-Aware Adaptive RAG (EA-RAG)** pipeline includes:

1. **Query-aware retrieval:** classify each query into a coarse intent (definition, risk factor, symptom/condition, diagnostic test, treatment, or comparison) and adjust retrieval settings or query expansion accordingly. If intent classification is unreliable, use the standard retrieval path.
2. **Evidence score gate:** threshold the top lexical score, retry once below a higher cutoff, and abstain if the final score remains below the lower cutoff. This version does not measure passage agreement or direct entailment.
3. **Citation checks:** open-ended outputs use structural citation checks. MCQ citations are optional; any IDs included must match retrieved chunks. Neither check verifies semantic support.

Adaptive retrieval, corrective retrieval, and selective prediction are established research areas. The project does not claim algorithmic novelty for these mechanisms. Its candidate contribution is the cardiology-oriented paired evaluation and its empirical coverage/robustness findings; the current test shows no significant selective-accuracy improvement and no P5 abstentions. See the experiment report for related work and a paper-safe contribution statement.

## 3. Scope and non-goals

### In scope
- Cardiology-focused question answering over a small, documented public-source corpus.
- Pre-chunked D4 loading, lexical retrieval, optional dense retrieval code, answer generation, citation checks, evidence gating, retry, and abstention.
- A baseline RAG implementation and an EA-RAG variant evaluated on the same questions.
- Reproducible metrics, error analysis, configuration, logs, and a small demo interface.

### Out of scope
- Clinical diagnosis or treatment recommendations for real patients.
- Fine-tuning an 8B model or reproducing the paper's 27k reasoning dataset.
- Access to private hospital records or protected health information.
- Claims of clinical validation, safety, or equivalence to a clinician.
- Fabricating results, datasets, citations, or comparisons.

## 4. High-level architecture

```text
                 ┌─────────────────────────────┐
                 │ Public cardiology documents│
                 └──────────────┬──────────────┘
                                v
                 ┌─────────────────────────────┐
                 │ Ingestion + provenance      │
                 │ parse / clean / metadata    │
                 └──────────────┬──────────────┘
                                v
                 ┌─────────────────────────────┐
                 │ Chunking + embedding        │
                 │ stable chunk IDs            │
                 └──────────────┬──────────────┘
                                v
                 ┌─────────────────────────────┐
                 │ Local vector index          │
                 └─────────────────────────────┘

 User question ──> Input validation
                        |
                        v
                 Query intent / normalization
                        |
                        v
                 First-pass retrieval
                        |
                        v
                 Evidence sufficiency gate ── insufficient ──> Query rewrite + retry
                        |                                          |
                     sufficient                              still insufficient
                        |                                          |
                        v                                          v
                 Grounded answer + citations                 Abstention response
                        |
                        v
                 Citation / output validation
                        |
                        v
                 Answer, sources, confidence signals, trace

 Evaluation runner: baseline top-k RAG vs EA-RAG on identical held-out questions
```

## 5. Components and contracts

### 5.1 Document ingestion
Each source must be public and permitted for the intended use. Store source title, publisher/organization, URL, access date, document date/version when available, and license/usage notes. Prefer authoritative guideline or public-health sources. Do not silently mix outdated and current guidance. Start with a small corpus (roughly 10–30 documents) and document inclusion criteria.

Output record:
```json
{
  "document_id": "stable-source-id",
  "title": "Source title",
  "publisher": "Organization",
  "url": "https://...",
  "published_date": "YYYY-MM-DD or null",
  "accessed_date": "YYYY-MM-DD",
  "license_note": "usage notes",
  "text": "cleaned text"
}
```

### 5.2 Chunking and indexing
- Preserve heading hierarchy and source metadata.
- Start with 400–700 token chunks and 60–100 token overlap; tune only on development data.
- Assign stable IDs such as `document_id:section:chunk_number`.
- Use a local vector store (FAISS or Chroma) behind a small retrieval interface.
- Persist the index and the exact corpus manifest; never rebuild an evaluation index from untracked sources.

### 5.3 Query processing
Return normalized query and one intent label. Use a deterministic keyword/rule baseline first; an LLM classifier may be optional. Avoid requiring an external model for the pipeline to function.

Intent labels: `definition`, `risk_factor`, `symptom_condition`, `test`, `treatment`, `comparison`, `other`.

### 5.4 Retrieval
Baseline retriever: vector similarity top-k with fixed k (default 4 or 5).
EA-RAG retriever: intent-aware query normalization/expansion, initial top-k retrieval, evidence checks, and at most one retry using a rewritten/expanded query. Keep the baseline and EA-RAG identical in corpus, generator, prompt, and evaluation set.

### 5.5 Evidence sufficiency gate
Implement transparent, inspectable heuristics first:
- Similarity threshold calibrated on development data, not test data.
- Minimum number of usable passages, configurable.
- Optional lexical overlap / key-entity coverage between question and passages.
- Agreement check: identify whether the top passages support compatible statements or contain material conflicts.
- Retry only once to bound latency.
- Abstain when evidence remains insufficient or conflicting.

Do not describe heuristic scores as calibrated clinical confidence. Expose component signals in logs and the UI.

### 5.6 Answer generation
Use a provider abstraction so the project can run with a configured API model or a local model. Do not hard-code secrets. If no LLM credentials are available, provide a deterministic extractive/demo fallback and label it as such.

Prompt requirements:
- Answer only from supplied context.
- Cite factual statements with stable chunk IDs.
- Explicitly say when the evidence is insufficient or conflicting.
- Separate source statements from cautious synthesis.
- Never claim to diagnose a user or replace a clinician.
- For emergencies or personal symptoms, display a short safety notice directing the user to local emergency services/qualified care rather than attempting triage.

### 5.7 Citation validation
Check that every cited chunk ID exists in the retrieved set and that at least one citation is present for each answer paragraph/claim group. This is a structural check, not proof that the cited source semantically entails the claim. Record failures for evaluation.

### 5.8 Interface
A minimal Streamlit UI is sufficient:
- Question input.
- Answer or abstention.
- Retrieved source snippets with title, date, and URL.
- Retrieval/evidence signals and retry status.
- Notice: research prototype, not for medical use.

### 5.9 Evaluation runner
Run baseline RAG and EA-RAG on exactly the same frozen questions, corpus, model settings, and decoding parameters. Cache model responses where allowed. Store raw outputs and metrics as versioned JSONL/CSV.

## 6. Data and benchmark strategy

- Create a modest question set from public cardiology educational/QA resources. Record each item’s source, correct answer or reference notes, category, and split.
- Aim for 60–120 questions if time allows; a smaller pilot is acceptable if transparently reported.
- Separate development questions from the final test questions before tuning thresholds/prompts.
- Avoid near-duplicate questions across splits and check for benchmark contamination where feasible.
- If using existing datasets, follow their licenses and rules; do not redistribute data where prohibited.
- Do not use the 49 private clinical cases from the baseline paper; they are not supplied here.
- If a trusted answer key cannot be established, evaluate evidence retrieval and citation validity rather than claiming diagnostic accuracy.

## 7. Metrics

Primary metrics:
- **Answer correctness:** accuracy or rubric-based correctness where a reliable answer key exists.
- **Citation validity:** fraction of citations that refer to retrieved chunk IDs.
- **Evidence support rate:** proportion of answer claims judged supported by cited source text (manual review on a sample is acceptable; report rubric and reviewer count).
- **Abstention quality:** abstention rate, unsupported-answer rate, and coverage-risk curve where feasible.
- **Retrieval quality:** Recall@k or Hit@k against manually tagged relevant source chunks on a small subset.

Secondary metrics:
- Mean/p50 latency, token usage or API cost if available.
- Retry rate and average retrieved passages.
- Category-level results and error taxonomy.

Report denominators, confidence intervals where useful, and failures. A high citation-validity score alone does not mean claims are true.

## 8. Experimental design

Compare:
1. **No-RAG baseline** (optional, only if the same model/provider can be run fairly).
2. **Fixed top-k RAG** (primary baseline).
3. **EA-RAG** (proposed method).
4. **EA-RAG without abstention** (ablation).
5. **EA-RAG without query adaptation/retry** (ablation).

Keep the corpus, generation model, prompt, and question set constant wherever possible. Tune thresholds only on the development split. Include at least one small qualitative error analysis table.

## 9. Reliability, privacy, and safety
- Public educational corpus only; no personal patient data.
- Do not present outputs as medical advice.
- Include an emergency disclaimer, but do not attempt to determine whether an individual is experiencing an emergency.
- Log no unnecessary personal data; avoid sending sensitive text to third-party APIs.
- Never log API keys.
- All outputs must be framed as experimental research results, not clinical performance guarantees.

## 10. Definition of done
- Reproducible ingestion and index build.
- Baseline RAG and EA-RAG both runnable.
- Evaluation script produces machine-readable outputs and metrics.
- At least one ablation and documented error analysis.
- UI shows sources and abstention state.
- Documentation states limitations and reports only measured results.
