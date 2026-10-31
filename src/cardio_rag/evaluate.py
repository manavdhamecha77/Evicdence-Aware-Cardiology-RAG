import json
import hashlib
import statistics
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import QAPipeline
from .generation import ExtractiveGenerator


def load_questions(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed question at {path}:{line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"Question at {path}:{line_number} must have non-empty question text")
            options = record.get("options")
            if options is not None:
                if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"} or not all(isinstance(v, str) for v in options.values()):
                    raise ValueError(f"MCQ options at {path}:{line_number} must contain text fields A, B, C, and D")
                answer = record.get("answer")
                if not isinstance(answer, str) or answer.strip().upper() not in {"A", "B", "C", "D"}:
                    raise ValueError(f"MCQ answer at {path}:{line_number} must be one of A, B, C, or D")
            records.append(record)
    return records


def evaluate(
    pipeline: QAPipeline,
    questions: list[dict[str, Any]],
    output_dir: Path,
    run_id: str = "local",
    sample_size: int | None = None,
    seed: int = 17,
    variants: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if sample_size is not None:
        if sample_size < 1 or sample_size > len(questions):
            raise ValueError("sample_size must be between 1 and the number of questions")
        rng = random.Random(seed)
        questions = sorted(rng.sample(questions, sample_size), key=lambda row: str(row.get("id", row.get("question_id", ""))))
    if any(question.get("options") for question in questions) and isinstance(pipeline.generator, ExtractiveGenerator):
        raise ValueError("MCQ evaluation requires a configured language model; the extractive fallback cannot select options")
    all_variants = ("baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation")
    variants = variants or all_variants
    if not variants or any(variant not in all_variants for variant in variants):
        raise ValueError(f"variants must be selected from {all_variants}")
    started = time.perf_counter()
    records = []
    raw_path = output_dir / f"{run_id}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for question in questions:
            for variant in variants:
                record = pipeline.answer(
                    question["question"], variant,
                    str(question.get("question_id", question.get("id", ""))),
                    question.get("reference_answer", question.get("answer")),
                    question.get("relevant_chunk_ids", []),
                    options=question.get("options"),
                    evaluation_mode=True,
                ).to_dict()
                records.append(record)
                raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                raw_file.flush()
    metrics = summarize(records, (time.perf_counter() - started) * 1000)
    (output_dir / f"{run_id}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config = pipeline.config
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(config.corpus_path),
        "corpus_sha256": hashlib.sha256(config.corpus_path.read_bytes()).hexdigest() if config.corpus_path.exists() else None,
        "source_manifest": str(config.source_manifest_path) if config.source_manifest_path else None,
        "source_manifest_sha256": hashlib.sha256(config.source_manifest_path.read_bytes()).hexdigest() if config.source_manifest_path and config.source_manifest_path.exists() else None,
        "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "questions_count": len(questions),
        "systems": list(variants),
        "generator": pipeline.generator.provider if pipeline.generator else "configuration_error",
        "model": pipeline.generator.model if pipeline.generator else None,
        "retrieval": config.retrieval_backend,
        "embedding_model": config.embedding_model if config.retrieval_backend == "dense" else None,
        "embedding_revision": config.embedding_revision if config.retrieval_backend == "dense" else None,
        "embedding_cache_key": getattr(pipeline.index, "cache_key", None),
        "config": {"top_k": config.top_k, "min_score": config.min_score, "retry_min_score": config.retry_min_score, "dense_min_score": config.dense_min_score, "dense_retry_min_score": config.dense_retry_min_score, "min_passages": config.min_passages, "chunk_words": config.chunk_words, "overlap_words": config.overlap_words, "seed": config.seed, "include_options_in_query": config.include_options_in_query, "sample_size": sample_size, "sample_seed": seed},
    }
    (output_dir / f"{run_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def summarize(records: list[dict[str, Any]], total_latency_ms: float = 0.0) -> dict[str, Any]:
    metrics: dict[str, Any] = {"record_count": len(records), "total_latency_ms": total_latency_ms, "systems": {}}
    for variant in ("baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"):
        rows = [row for row in records if row["variant"] == variant]
        errors = [row for row in rows if row.get("metadata", {}).get("run_error")]
        valid_rows = [row for row in rows if not row.get("metadata", {}).get("run_error")]
        citation_rows = [row for row in valid_rows if not row["abstained"]]
        labeled = [row for row in valid_rows if row["relevant_chunk_ids"]]
        mcq_rows = [row for row in valid_rows if row.get("predicted_answer") is not None or (row.get("reference_answer") and row.get("metadata", {}).get("task") == "mcq")]
        mcq_answered = [row for row in mcq_rows if not row["abstained"]]
        open_qa_rows = [row for row in valid_rows if row.get("reference_answer") is not None and row.get("metadata", {}).get("task") != "mcq"]
        abstentions = sum(row["abstained"] for row in valid_rows)
        metrics["systems"][variant] = {
            "count": len(valid_rows),
            "run_error_count": len(errors),
            "run_error_rate": len(errors) / len(rows) if rows else None,
            "abstention_rate": abstentions / len(valid_rows) if valid_rows else None,
            "coverage": (len(valid_rows) - abstentions) / len(valid_rows) if valid_rows else None,
            "accuracy": sum(bool(row.get("correct")) for row in mcq_rows) / len(mcq_rows) if mcq_rows else None,
            "accuracy_denominator": len(mcq_rows),
            "selective_accuracy": sum(bool(row.get("correct")) for row in mcq_answered) / len(mcq_answered) if mcq_answered else None,
            "invalid_answer_rate": sum(row.get("predicted_answer") is None for row in mcq_answered) / len(mcq_answered) if mcq_answered else None,
            "citation_validity": sum(row["citation_valid"] for row in citation_rows) / len(citation_rows) if citation_rows else None,
            "citation_validity_denominator": len(citation_rows),
            "hit_at_k": sum(bool(set(row["retrieved_chunk_ids"]) & set(row["relevant_chunk_ids"])) for row in labeled) / len(labeled) if labeled else None,
            "hit_at_k_denominator": len(labeled),
            "reference_substring_accuracy": sum(row["reference_answer"].casefold() in row["answer"].casefold() for row in open_qa_rows) / len(open_qa_rows) if open_qa_rows else None,
            "reference_answer_denominator": len(open_qa_rows),
            "mean_latency_ms": statistics.mean(row.get("metadata", {}).get("primary_latency_ms", row["latency_ms"]) for row in valid_rows) if valid_rows else None,
            "p50_latency_ms": statistics.median(row.get("metadata", {}).get("primary_latency_ms", row["latency_ms"]) for row in valid_rows) if valid_rows else None,
            "retry_rate": sum(row["retry_count"] > 0 for row in valid_rows) / len(valid_rows) if valid_rows else None,
            "unsupported_answer_rate": None,
            "evidence_support_rate": None,
        }
    return metrics
