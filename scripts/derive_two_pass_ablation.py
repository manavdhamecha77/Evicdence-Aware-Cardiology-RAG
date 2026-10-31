"""Build the four-arm MCQ comparison from fixed and full EA-RAG run logs."""

import argparse
import copy
import json
import math
from pathlib import Path

from cardio_rag.evaluate import summarize


ABSTENTION_TEXT = (
    "The available indexed sources are insufficient to answer this question reliably. "
    "Research prototype for educational use only; not medical advice or a clinical decision system."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derive(baseline: list[dict], ea_rows: list[dict]) -> list[dict]:
    by_id = {row["question_id"]: row for row in ea_rows}
    if len(by_id) != len(ea_rows) or {row["question_id"] for row in baseline} != set(by_id):
        raise ValueError("Fixed and EA-RAG runs must contain the same unique question IDs")

    records = [copy.deepcopy(row) for row in baseline]
    for row in records:
        row["variant"] = "baseline"

    for source in ea_rows:
        full = copy.deepcopy(source)
        full["variant"] = "ea_rag"
        records.append(full)

        no_abstain = copy.deepcopy(source)
        no_abstain["variant"] = "ea_rag_no_abstention"
        if source["abstained"]:
            candidate = source.get("metadata", {}).get("counterfactual_no_abstention")
            if not candidate:
                raise ValueError(f"Missing no-abstention counterfactual for {source['question_id']}")
            no_abstain.update(
                answer=candidate["answer"],
                predicted_answer=candidate["predicted_answer"],
                correct=candidate["correct"],
                abstained=False,
                abstention_reason=None,
            )
            no_abstain.setdefault("metadata", {})["primary_latency_ms"] = candidate.get("latency_ms", source["latency_ms"])
        records.append(no_abstain)

        no_adaptation = copy.deepcopy(source)
        no_adaptation["variant"] = "ea_rag_no_adaptation"
        initial_sufficient = bool(source.get("metadata", {}).get("initial_sufficient"))
        if not initial_sufficient:
            no_adaptation.update(
                answer=ABSTENTION_TEXT,
                predicted_answer=None,
                correct=False,
                abstained=True,
                abstention_reason="Initial retrieval did not meet the evidence threshold.",
            )
        elif source.get("retry_count", 0):
            candidate = source.get("metadata", {}).get("counterfactual_no_adaptation")
            if not candidate:
                raise ValueError(f"Missing no-adaptation counterfactual for {source['question_id']}")
            no_adaptation.update(
                answer=candidate["answer"],
                predicted_answer=candidate["predicted_answer"],
                correct=candidate["correct"],
                abstained=False,
                abstention_reason=None,
            )
            no_adaptation.setdefault("metadata", {})["primary_latency_ms"] = candidate.get("latency_ms", source["latency_ms"])
        no_adaptation["retry_count"] = 0
        records.append(no_adaptation)
    return records


def paired_mcnemar(records: list[dict], left: str, right: str) -> dict[str, float | int]:
    by_arm = {
        arm: {row["question_id"]: bool(row.get("correct")) for row in records if row["variant"] == arm}
        for arm in (left, right)
    }
    if by_arm[left].keys() != by_arm[right].keys():
        raise ValueError(f"Arms {left} and {right} do not have matching question IDs")
    pairs = [(by_arm[left][qid], by_arm[right][qid]) for qid in by_arm[left]]
    left_only = sum(a and not b for a, b in pairs)
    right_only = sum(b and not a for a, b in pairs)
    discordant = left_only + right_only
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / (2**discordant) if discordant else 1.0
    return {
        "n": len(pairs),
        "left_correct_right_incorrect": left_only,
        "left_incorrect_right_correct": right_only,
        "two_sided_exact_p": min(1.0, 2 * tail),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ea-rag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = derive(read_jsonl(args.baseline), read_jsonl(args.ea_rag))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    metrics = summarize(records)
    metrics["paired_mcnemar"] = {
        "baseline_vs_ea_rag": paired_mcnemar(records, "baseline", "ea_rag"),
        "ea_rag_vs_no_abstention": paired_mcnemar(records, "ea_rag", "ea_rag_no_abstention"),
        "ea_rag_vs_no_adaptation": paired_mcnemar(records, "ea_rag", "ea_rag_no_adaptation"),
    }
    args.output.with_name(args.output.stem + "_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"records": len(records), "metrics_path": str(args.output.with_name(args.output.stem + "_metrics.json"))}, indent=2))


if __name__ == "__main__":
    main()
