"""Prepare provisional, source-grouped NHLBI QA splits from prepared_data/.

All relevance suggestions are machine-generated candidates. They are not gold
labels and must not be treated as benchmark judgements without review.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardio_rag.retrieval import LexicalIndex  # noqa: E402
from cardio_rag.schemas import Chunk  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def clean_text(value: str | None) -> str:
    value = html.unescape(value or "")
    value = value.replace("\u00a0", " ").replace("\ufeff", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([?.!,;:])", r"\1", value)
    return value


def clean_question(value: str | None) -> str:
    question = clean_text(value)
    question = re.sub(r"\bWhat is\s+\(are\)\s+", "What is ", question, flags=re.IGNORECASE)
    question = re.sub(r"\s*\?+\s*$", "?", question)
    return question


def canonical(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def grouped_split(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    grouped: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[row["document_id"]] += 1

    rng = random.Random(seed)
    document_ids = list(grouped)
    rng.shuffle(document_ids)
    document_ids.sort(key=lambda doc_id: grouped[doc_id], reverse=True)

    names = ("train", "dev", "test")
    ratios = {"train": 0.70, "dev": 0.15, "test": 0.15}
    assigned = {name: 0 for name in names}
    total = sum(grouped.values())
    mapping: dict[str, str] = {}
    for position, doc_id in enumerate(document_ids):
        # Seed one complete source document into each split before balancing size.
        if position < len(names):
            chosen = names[position]
        else:
            chosen = max(
                names,
                key=lambda name: (ratios[name] * total - assigned[name]) / ratios[name],
            )
        mapping[doc_id] = chosen
        assigned[chosen] += grouped[doc_id]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "prepared_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared_data" / "processed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    source_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(source_dir / "download_manifest.csv")
    downloaded = {
        row["document_id"]: row
        for row in manifest_rows
        if row.get("status", "").startswith("downloaded")
    }

    chunks: list[Chunk] = []
    for row in read_csv(source_dir / "chunks.csv"):
        source = downloaded.get(row["document_id"])
        if source is None:
            continue
        chunks.append(
            Chunk(
                chunk_id=clean_text(row["chunk_id"]),
                document_id=clean_text(row["document_id"]),
                title=clean_text(row["topic"]),
                publisher="National Heart, Lung, and Blood Institute (NHLBI)",
                url=clean_text(source.get("final_url") or row.get("source_url")),
                accessed_date="unknown",
                license_note=(
                    "Usage terms are not recorded in the supplied prepared_data manifest; "
                    "verify before redistribution."
                ),
                section=clean_text(row["topic"]),
                text=clean_text(row["text"]),
            )
        )

    chunks_by_document: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk.chunk_id)
    index = LexicalIndex(chunks)

    candidates: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    exclusion_counts: dict[str, int] = defaultdict(int)
    for row in read_csv(source_dir / "nhlbi_cardiology_questions.csv"):
        doc_id = clean_text(row.get("document_id"))
        question = clean_question(row.get("question"))
        answer = clean_text(row.get("reference_answer"))
        if doc_id not in downloaded or not chunks_by_document.get(doc_id):
            exclusion_counts["source_not_downloaded_or_chunked"] += 1
            continue
        if not question or not answer:
            exclusion_counts["empty_question_or_answer"] += 1
            continue
        question_key = canonical(question)
        if question_key in seen_questions:
            exclusion_counts["duplicate_question"] += 1
            continue
        seen_questions.add(question_key)
        ranked = index.retrieve(question, args.top_k)
        candidates.append(
            {
                "question_id": clean_text(row.get("question_id")) or f"nhlbi-{doc_id}-{len(candidates)+1}",
                "document_id": doc_id,
                "topic": clean_text(row.get("topic")),
                "question_type": clean_text(row.get("question_type")),
                "question": question,
                "reference_answer": answer,
                "source_url": clean_text(downloaded[doc_id].get("final_url") or row.get("source_url")),
                "split": "",
                "labels_status": "unreviewed_candidate",
                "candidate_retrieved_chunk_ids": [item.chunk.chunk_id for item in ranked],
                "relevant_chunk_ids": [],
                "review_note": (
                    "Machine-generated candidate. Verify question relevance, reference answer, "
                    "source capture, and supporting passages before evaluation."
                ),
            }
        )

    split_by_doc = grouped_split(candidates, args.seed)
    for row in candidates:
        row["split"] = split_by_doc[row["document_id"]]

    def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "chunks.jsonl", [chunk.__dict__ for chunk in chunks])
    write_jsonl(output_dir / "all_candidates.jsonl", candidates)
    for split_name in ("train", "dev", "test"):
        split_records = [row for row in candidates if row["split"] == split_name]
        write_jsonl(output_dir / f"{split_name}.jsonl", split_records)

    review_columns = (
        "question_id", "document_id", "topic", "question_type", "question",
        "reference_answer", "source_url", "split", "candidate_retrieved_chunk_ids",
        "include_in_evaluation", "answerable_from_corpus", "relevant_chunk_ids", "reviewer_notes",
    )
    with (output_dir / "review_template.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=review_columns)
        writer.writeheader()
        for row in candidates:
            writer.writerow({
                **{key: row.get(key, "") for key in review_columns},
                "candidate_retrieved_chunk_ids": ";".join(row["candidate_retrieved_chunk_ids"]),
            })

    report = {
        "status": "provisional; not benchmark-ready",
        "seed": args.seed,
        "split_unit": "document_id; all questions from a source document stay together",
        "requested_split_ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "question_count": len(candidates),
        "document_count": len(split_by_doc),
        "chunk_count": len(chunks),
        "split_counts": {
            split_name: {
                "questions": sum(row["split"] == split_name for row in candidates),
                "documents": sum(value == split_name for value in split_by_doc.values()),
            }
            for split_name in ("train", "dev", "test")
        },
        "excluded": dict(exclusion_counts),
        "relevance_labels": "relevant_chunk_ids intentionally empty; candidate_retrieved_chunk_ids are suggestions only",
        "review_template": "review_template.csv includes blank human-review fields",
        "limitations": [
            "The topic-filtered questions have not been manually reviewed for cardiology relevance.",
            "NHLBI reference answers may not match the captured webpage version.",
            "Downloaded pages are marked needs_review and may contain navigation or extraction noise.",
            "Source usage terms are absent from the supplied manifest and must be verified.",
            "No answerability or human evidence-support labels have been established.",
        ],
    }
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
