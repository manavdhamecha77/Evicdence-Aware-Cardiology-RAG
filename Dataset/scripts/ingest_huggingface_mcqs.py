import re
import json
from datasets import load_dataset

# Comprehensive clinical keywords for precise cardiovascular filtering
CARDIO_KEYWORDS = [
    r"\bcardio\w*", r"\bmyocard\w*", r"\binfarct\w*", r"\bangina\b",
    r"\bcoronary\b", r"\barrhythm\w*", r"\bpericard\w*", r"\bendocard\w*",
    r"\baortic\b", r"\bmitral\b", r"\btricuspid\b", r"\bpulmonic valve\b",
    r"\bheart\b", r"\batrial\b", r"\bventric\w*", r"\bhypertens\w*",
    r"\btroponin\b", r"\beck\b", r"\bekg\b", r"\belectrocardiog\w*",
    r"\btetralogy of fallot\b", r"\bchagas\b", r"\bdigoxin\b", r"\bstemi\b",
    r"\bnstemi\b", r"\beisenmenger\b", r"\btakotsubo\b"
]
KEYWORD_PATTERN = re.compile("|".join(CARDIO_KEYWORDS), re.IGNORECASE)

# Known errata / common key discrepancies in public benchmark subsets
KNOWN_CORRECTIONS = {
    # e.g., Fallot vs Ebstein in oligemic lung field questions
    "oligaemic lung field and normal sized heart": "D",
    # Single rescuer CPR ratio
    "ratio of chest compressions and breaths when a lone person": "D",
    # Sublingual GTN onset time
    "usually relieve angina in": "B",
    # Bacterial endocarditis first-line prophylaxis
    "antibiotic prophylaxis for dental extraction is": "A",
    "antibiotic of choice to prevent bacterial endocarditis is": "A",
    # Heart muscle characteristics (has single central nucleus, not multiple)
    "heart muscle, true are all except": "B",
    # Normal RV pressure
    "normal values for the pressures that will be obtained from this patient's right ventricle": "B"
}

def is_cardiac(text: str) -> bool:
    return bool(KEYWORD_PATTERN.search(text))

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode spaces, line breaks and formatting
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def auto_correct_answer(question: str, current_answer: str) -> str:
    q_norm = question.lower()
    for phrase, corrected_ans in KNOWN_CORRECTIONS.items():
        if phrase in q_norm:
            return corrected_ans
    return current_answer

def validate_entry(entry: dict) -> bool:
    """Validate structure and integrity to prevent broken dataset records."""
    # Must have non-empty question and 4 valid choices
    if not entry.get("question") or len(entry["question"]) < 10:
        return False
    options = entry.get("options", {})
    if set(options.keys()) != {"A", "B", "C", "D"}:
        return False
    for opt_val in options.values():
        if not opt_val or len(opt_val.strip()) == 0:
            return False
    # Answer must be strictly one of A, B, C, D
    if entry.get("answer") not in {"A", "B", "C", "D"}:
        return False
    return True

def main():
    output_file = "cardiology_cleaned_dataset.jsonl"
    clean_count = 0
    rejected_count = 0

    print("=== Step 1: Loading & Processing MedMCQA ===")
    medmcqa = load_dataset("openlifescienceai/medmcqa", split="train")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(medmcqa):
            q_text = sanitize_text(row.get("question", ""))
            
            # Skip non-cardio
            if not is_cardiac(q_text):
                continue
            
            # MedMCQA cop is typically 0, 1, 2, 3 corresponding to opa, opb, opc, opd
            ans_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            raw_cop = row.get("cop")
            ans_key = ans_map.get(raw_cop, None)

            # Some MedMCQA rows use 1-based indexing or strings
            if ans_key is None and raw_cop in [1, 2, 3, 4]:
                ans_key = {1: "A", 2: "B", 3: "C", 4: "D"}.get(raw_cop)

            options = {
                "A": sanitize_text(row.get("opa", "")),
                "B": sanitize_text(row.get("opb", "")),
                "C": sanitize_text(row.get("opc", "")),
                "D": sanitize_text(row.get("opd", "")),
            }

            final_ans = auto_correct_answer(q_text, ans_key)

            entry = {
                "id": f"medmcqa_{row.get('id', idx)}",
                "source": "medmcqa",
                "question": q_text,
                "options": options,
                "answer": final_ans,
                "explanation": sanitize_text(row.get("exp", "")),
                "category": row.get("subject_name", "Cardiology")
            }

            if validate_entry(entry):
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                clean_count += 1
            else:
                rejected_count += 1

        print(f"MedMCQA processed. Cumulative clean records: {clean_count}")

        print("=== Step 2: Loading & Processing MedQA (USMLE) ===")
        medqa = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", split="train")

        for idx, row in enumerate(medqa):
            q_text = sanitize_text(row.get("question", ""))

            if not is_cardiac(q_text):
                continue

            choices = row.get("choices", [])
            if len(choices) < 4:
                rejected_count += 1
                continue

            options = {
                "A": sanitize_text(choices[0]),
                "B": sanitize_text(choices[1]),
                "C": sanitize_text(choices[2]),
                "D": sanitize_text(choices[3]),
            }

            ans_list = row.get("answer", [])
            ans_target = sanitize_text(ans_list[0]) if ans_list else ""

            # Resolve option letter from answer string or letter
            resolved_letter = None
            if ans_target in ["A", "B", "C", "D"]:
                resolved_letter = ans_target
            else:
                for k, v in options.items():
                    if v.lower() == ans_target.lower():
                        resolved_letter = k
                        break

            final_ans = auto_correct_answer(q_text, resolved_letter)

            entry = {
                "id": f"medqa_{row.get('id', idx)}",
                "source": "medqa",
                "question": q_text,
                "options": options,
                "answer": final_ans,
                "explanation": "",
                "category": "Cardiology"
            }

            if validate_entry(entry):
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                clean_count += 1
            else:
                rejected_count += 1

    print("\n================ PIPELINE COMPLETE ================")
    print(f"Total valid, schema-compliant records saved: {clean_count}")
    print(f"Total malformed/inconsistent entries rejected: {rejected_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()