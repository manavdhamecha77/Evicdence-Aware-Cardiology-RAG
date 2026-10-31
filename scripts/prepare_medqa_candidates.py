"""Prepare a review-only cardiology candidate set from MedQA-USMLE's US test split.

Keyword matching is deliberately only a candidate-generation step. It is not a
validated specialty classifier; manually review the output before evaluation.
The script does not read or ingest the accompanying textbook files.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


CARDIOLOGY_TERMS = re.compile(
    r"\b(?:cardio\w*|heart|myocard\w*|coronary|angina|atrial|ventric\w*|"
    r"aortic|mitral|tricuspid|pulmonic valve|valv\w*|arrhythm\w*|"
    r"tachycard\w*|bradycard\w*|fibrillation|flutter|pericard\w*|"
    r"endocard\w*|electrocardi\w*|\b(?:ecg|ekg)\b|ejection fraction|"
    r"cardiac output|heart failure|hypertension|atheroscleros\w*|"
    r"myoglobin|troponin|cardiomyopath\w*|cardiac catheter\w*|"
    r"congenital heart|cardiogenic shock|pulse pressure)\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("question"), str):
                raise ValueError(f"Invalid question record at {path}:{line_number}")
            records.append(record)
    return records


def to_candidate(record: dict[str, Any], position: int) -> dict[str, Any]:
    options = record.get("options", {})
    if isinstance(options, dict):
        option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    else:
        option_text = str(options)
    question = record["question"].strip()
    if option_text:
        question = f"{question}\n\nAnswer choices:\n{option_text}"
    answer = record.get("answer")
    return {
        "question_id": f"medqa-us-test-{position:04d}",
        "question": question,
        "reference_answer": answer if isinstance(answer, str) else None,
        "answer_idx": record.get("answer_idx"),
        "dataset": "MedQA-USMLE",
        "source_split": "US/test.jsonl",
        "meta_info": record.get("meta_info"),
        "topic_review_status": "keyword_candidate_requires_manual_review",
        "relevant_chunk_ids": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Defaults to data/MedQA-USMLE/questions/US/<split>.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to data/processed/medqa_us_<split>_cardio_candidates.jsonl",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    args = parser.parse_args()

    input_path = args.input or Path(f"data/MedQA-USMLE/questions/US/{args.split}.jsonl")
    output_path = args.output or Path(f"data/processed/medqa_us_{args.split}_cardio_candidates.jsonl")
    records = load_jsonl(input_path)
    candidates = [
        to_candidate(record, position)
        for position, record in enumerate(records, start=1)
        if CARDIOLOGY_TERMS.search(record["question"])
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "test_questions_read": len(records),
                "cardiology_keyword_candidates": len(candidates),
                "output": str(output_path),
                "warning": "Candidates require manual cardiology and licensing review; not a validated benchmark split.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
