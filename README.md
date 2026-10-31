# Evidence-Aware Adaptive RAG for Cardiology QA

A local research prototype comparing fixed top-k retrieval with evidence-gated retrieval, one bounded retry, and abstention for cardiology educational questions.

**Current experiment:** D2 test (609 items), P5 stress test (21 items), and a D1 retrieval-only audit are complete. The full report includes methods, artifacts, results, limits, and research framing: [docs/REPORT.md](docs/REPORT.md).

## Current implementation

- Source manifest validation and ingestion from local UTF-8 text or Markdown files.
- Stable heading-aware chunks with source metadata and JSONL persistence.
- Deterministic BM25-style lexical retrieval and optional CPU sentence-transformer cosine retrieval with a content/model-keyed embedding cache.
- Fixed top-k baseline and EA-RAG with keyword intent routing, query expansion, a configurable evidence threshold, one retry, and abstention.
- A deterministic extractive generator, structural citation validation, JSONL evaluation records, aggregate JSON metrics, and run manifests.
- Optional OpenAI Chat Completions generation with a grounded prompt; API/network errors are returned as explicit failed runs.
- Ablations without abstention and without adaptation/retry, plus a minimal Streamlit interface.

Frozen D1-D6 benchmark data and D4 pre-chunked corpora are under `data/`. The saved experiment report documents the current results and evaluation limits.

The local `data/` folder contains frozen D1-D6 artifacts and D4 pre-chunked guideline corpora. The currently reported evaluation used hierarchical D4 and D2 MCQs. It used lexical retrieval and Ollama Cloud `gemma4:31b`; it did not use embeddings or a vector database. See the experiment report for counts, hashes, metrics, and limits. Check source and dataset licenses before external processing or redistribution.

## Local MedQA-USMLE question data

An external MedQA-USMLE copy may be placed under `data/MedQA-USMLE/`; that directory and `data/archive.zip` are ignored by Git. Its US `dev.jsonl` and `test.jsonl` splits are a possible secondary set of USMLE-style multiple-choice questions, not a consumer cardiology FAQ benchmark. Generate cardiology keyword candidates for manual review while preserving the official split:

```powershell
python scripts/prepare_medqa_candidates.py
python scripts/prepare_medqa_candidates.py --split dev
```

The script reads one official split at a time, includes answer choices and reference answers, and writes a review-only candidate JSONL under `data/processed/`. Keyword matches are overinclusive and must be manually checked; the output is not a vetted cardiology set or a benchmark result. The normal evaluator does not consume this file automatically. Create a provenance manifest for the supplied English textbook extracts and ingest them locally with:

```powershell
python scripts/prepare_medqa_manifest.py
python -m cardio_rag.cli ingest --manifest data/processed/medqa_sources_manifest.json --output data/processed/medqa_chunks.jsonl
```

The MedQA paper describes its textbook collection as released under a research-use-only license agreement. Use only under the terms applicable to your copy; do not redistribute source text or derived passages. Preserve provenance and cite the dataset paper.

See [docs/TODO.md](docs/TODO.md) for the next research and implementation tasks.

## Setup (PowerShell)

Install the pinned dependencies manually into the existing root virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Ingestion, chunking, and lexical retrieval use the Python standard library. Dense retrieval uses `sentence-transformers` on CPU and caches vectors under `artifacts/indices`; the first run downloads the selected embedding model. `streamlit` runs the demo and `pytest` runs the included unit tests.

Generation defaults to the labeled extractive fallback. To use Ollama Cloud, copy `.env.example` to `.env`, then set `OLLAMA_API_KEY`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL=https://ollama.com/v1`. Ollama-compatible `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` names are also accepted. Process environment variables take precedence over `.env` values.

The request sends the question and retrieved public source passages to the configured endpoint. Never use personal health information. The key is read from `.env` or the environment and is not written to run artifacts. Restart the app after changing these settings.

## Add sources and build the corpus

Create one local `.txt` or `.md` file per public source. The manifest is a JSON array with required provenance fields; paths are relative to the manifest file:

```json
[
  {
    "document_id": "organization-topic-2026",
    "title": "Exact title of the source",
    "publisher": "Publishing organization",
    "url": "https://publisher.example/source",
    "published_date": "YYYY-MM-DD",
    "accessed_date": "YYYY-MM-DD",
    "license_note": "License or terms permitting this use",
    "path": "organization-topic.md"
  }
]
```

Record the exact source and its version/date in your manifest and local file. The tool does not fetch URLs or verify licenses. Then run:

```powershell
python -m cardio_rag.cli ingest
```

Chunk defaults are 550 whitespace-delimited words with 80 words of overlap. The generated `data/processed/chunks.jsonl` is ignored by Git; preserve your source files and manifest to reproduce it.

## Prepare the supplied NHLBI data

The optional `prepared_data/` folder contains NHLBI QA candidates, downloaded page text, and pre-split passages. Build app-compatible chunks and deterministic, document-grouped candidate splits with the existing uv environment:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --python .venv\Scripts\python.exe python scripts\prepare_nhlbi_evaluation.py
```

Outputs are written to `prepared_data/processed/`: `chunks.jsonl`, `all_candidates.jsonl`, `train.jsonl`, `dev.jsonl`, and `test.jsonl`, plus `review_template.csv` and a preprocessing report. All questions from a source document stay in the same split. Relevance suggestions are stored as `candidate_retrieved_chunk_ids`; `relevant_chunk_ids` intentionally remains empty until passages are reviewed. These candidate splits are not benchmark results until question relevance, source captures, reference answers, source terms, answerability, and supporting chunks are reviewed. See [prepared_data/dataset desc.md](prepared_data/dataset%20desc.md).

## Ask and evaluate

```powershell
python -m cardio_rag.cli ask "What is the definition of atrial fibrillation?" --variant ea_rag
python -m cardio_rag.cli evaluate --questions data/evaluation/dev.jsonl --run-id pilot
python -m cardio_rag.cli evaluate --questions data/evaluation/dev.jsonl --run-id pilot-dense --retrieval dense
```

### Run the D2 MCQ split on pre-chunked D4

`data/` contains pre-chunked corpora. Pass the D4 source metadata file so the loader can preserve each existing `chunk_id` and attach its provenance. For a local Ollama server, configure `OLLAMA_MODEL=llama3.1:8b` and `OLLAMA_BASE_URL=http://localhost:11434/v1`; no API key is needed for localhost. The extractive fallback cannot answer MCQs, so the evaluator stops rather than reporting fake accuracy if no language model is configured.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:PYTHONPATH = 'src'
$env:OLLAMA_MODEL = 'llama3.1:8b'
$env:OLLAMA_BASE_URL = 'http://localhost:11434/v1'
uv run --python .venv\Scripts\python.exe python -m cardio_rag.cli evaluate `
  --questions data\artifacts\D2_val_clean.jsonl `
  --corpus 'data\knowledge base\indexed chunks\D4_cardiology_kb_hierarchical.jsonl' `
  --source-manifest data\manifests\D4_sources.json `
  --retrieval dense --sample-size 50 --include-options-in-query `
  --output data\artifacts\runs --run-id d2-val-50
```

The evaluator scores MCQ letters with a strict `FINAL ANSWER: A` parser, reports accuracy and coverage, and disables the personal-symptom chat guard only for evaluation. The frozen D2 run used stem-only retrieval and thresholds selected from D2 validation score quantiles. D5 labels currently match only the fixed-window D4 IDs and are keyword-generated candidates, not human-judged relevance labels.

Lexical retrieval is the default and requires no model download. Dense retrieval defaults to `sentence-transformers/all-MiniLM-L6-v2`; set `CARDIO_RAG_EMBEDDING_MODEL` or pass `--embedding-model` to select another model. Set `CARDIO_RAG_EMBEDDING_REVISION` or pass `--embedding-revision` to pin a model revision. The initial dense evidence threshold is a configurable heuristic and must be tuned on development data only.

Question JSONL records require a non-empty `question`; optional fields include `question_id`, `reference_answer`, and `relevant_chunk_ids`. The evaluator runs baseline, EA-RAG, no-abstention, and no-adaptation variants on the same records. It writes per-question `artifacts/pilot.jsonl`, aggregate `artifacts/pilot_metrics.json`, and configuration `artifacts/pilot_manifest.json`. It calculates citation structure validity, Hit@k when relevance labels exist, a simple reference-answer substring rate when answers are provided, abstention, and latency. Substring match is not a clinical correctness metric. Unsupported-answer and semantic citation support rates require human review and are not inferred by this prototype.

Run the interface after building a corpus:

```powershell
python -m streamlit run src/cardio_rag/app.py
```

## Safety and research limits

This is an educational research prototype, not a medical device, diagnostic service, or substitute for professional care. The UI warns users not to enter personal health information and the pipeline returns a safety notice for recognized first-person symptom queries. The notice uses simple keyword matching and is not a triage system. Never use outputs for clinical decisions.

Retrieval scores are ranking signals, not calibrated confidence. Query intent and evidence sufficiency use transparent heuristics. The extractive fallback copies source sentences; citations are checked structurally and are not proof of semantic support. Current results are exploratory: the D2 selective-accuracy difference was not significant on the answered subset, P5 did not trigger abstention, and D1 diagnostic accuracy is unavailable. There is no automatic contradiction detector or clinician-validated benchmark.

## Research reference

Yang, Z. et al. “Cardiology-Chat: A Multi-LLMs Powered System for Cardiac Diagnostic Reasoning and Clinical Support.” *IEEE Journal of Translational Engineering in Health and Medicine*, 2026. DOI: [10.1109/JTEHM.2026.3668755](https://doi.org/10.1109/JTEHM.2026.3668755). This repository does not reproduce its training or evaluation and does not claim established novelty or clinical performance.
