"""Local evidence-gate coverage audit for D1 Stage 1 presentations."""

import json
import statistics
from pathlib import Path

from cardio_rag.config import Config
from cardio_rag.ingest import read_chunks
from cardio_rag.query_router import adapt_query, classify_intent
from cardio_rag.retrieval import LexicalIndex


ROOT = Path("data")
CASES = ROOT / "artifacts" / "D1_cases_staged.jsonl"
CORPUS = ROOT / "knowledge base" / "indexed chunks" / "D4_cardiology_kb_hierarchical.jsonl"
MANIFEST = ROOT / "manifests" / "D4_sources.json"
OUT = ROOT / "artifacts" / "d1_stage1_retrieval_audit.json"


def main() -> None:
    config = Config(corpus_path=CORPUS, source_manifest_path=MANIFEST)
    index = LexicalIndex(read_chunks(CORPUS, MANIFEST))
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for case in cases:
        query = case["stage_1_presentation"]
        intent = classify_intent(query)
        initial = index.retrieve(query, config.top_k)
        initial_top = initial[0].score if initial else 0.0
        retry_count = int(initial_top < config.retry_trigger_min_score)
        final = initial
        if retry_count:
            retry = index.retrieve(adapt_query(query, intent), config.top_k)
            merged = {item.chunk.chunk_id: item for item in initial}
            for item in retry:
                if item.chunk.chunk_id not in merged or item.score > merged[item.chunk.chunk_id].score:
                    merged[item.chunk.chunk_id] = item
            final = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.chunk_id))[:config.top_k]
        final_top = max((item.score for item in final), default=0.0)
        records.append({
            "case_id": case["id"],
            "year": case.get("year"),
            "initial_top1_score": initial_top,
            "retry_count": retry_count,
            "final_top1_score": final_top,
            "abstained": final_top < config.evidence_min_score,
            "retrieved_chunk_ids": [item.chunk.chunk_id for item in final],
        })

    scores = [row["initial_top1_score"] for row in records]
    summary = {
        "task": "D1 Stage 1 local retrieval/evidence-gate audit only",
        "case_count": len(records),
        "retrieval": "lexical",
        "abstention_threshold": config.evidence_min_score,
        "retry_threshold": config.retry_trigger_min_score,
        "initial_retry_count": sum(row["retry_count"] for row in records),
        "final_abstentions": sum(row["abstained"] for row in records),
        "coverage_after_retry": sum(not row["abstained"] for row in records) / len(records) if records else None,
        "initial_top1_score_median": statistics.median(scores) if scores else None,
        "diagnostic_accuracy": None,
        "accuracy_note": "No validated diagnosis matcher is configured; this audit does not score diagnoses.",
        "records": records,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
