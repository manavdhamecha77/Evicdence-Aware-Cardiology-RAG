import json
import re
import hashlib

INPUT_D2 = "artifacts/D2_test_clean.jsonl"
OUTPUT_P5 = "artifacts/D3_P5_stripped.jsonl"

CLINICAL_ANCHORS = [
    # Physical Exam Findings
    r"\bwater-hammer pulse\b", r"\bCorrigan'?s? pulse\b", r"\bpulsus paradoxus\b",
    r"\bpulsus alternans\b", r"\bpulsus parvus et tardus\b", r"\bJaneway lesions?\b",
    r"\bOsler(?:'s)? nodes?\b", r"\bRoth(?:'s)? spots?\b", r"\bsplinter hemorrhages?\b",
    r"\bfriction rub\b", r"\bopening snap\b", r"\bmid-systolic click\b",
    r"\b(?:third heart sound|S3 gallop|S3 sound)\b", r"\b(?:fourth heart sound|S4 gallop|S4 sound)\b",
    r"\bKussmaul(?:'s)? sign\b", r"\bLevine(?:'s)? sign\b", r"\bbounding pulses?\b",
    r"\bcannon (?:a|A) waves?\b", r"\bheave\b", r"\bthrill\b",

    # Pathological & Biopsy Features
    r"\bAschoff (?:bodies|nodules)\b", r"\bAnitschkow (?:cells|myocytes)\b",
    r"\b(?:biconvex\s+)?spindle-shaped vacuoles\b", r"\bcholesterol (?:clefts|emboli)\b",
    r"\bapple-green birefringence\b", r"\bCongo red staining\b",
    r"\bcontraction band necrosis\b", r"\bcurvilinear inclusion bodies\b",

    # Diagnostic Imaging Signatures
    r"\bboot-shaped heart\b", r"\begg-(?:shaped|on-a-string) heart\b",
    r"\bapical ballooning\b", r"\bwater-bottle (?:heart|sign)\b",
    r"\bfigure-of-3 sign\b", r"\bHampton(?:'s)? hump\b", r"\bWestermark(?:'s)? sign\b",
    r"\bmachine-like murmur\b", r"\bholosystolic murmur\b", r"\bdecrescendo murmur\b",

    # Key ECG Patterns
    r"\belectrical alternans\b", r"\bdelta waves?\b", r"\bshort PR interval\b",
    r"\btorsades de pointes\b", r"\bepsilon waves?\b", r"\bBrugada pattern\b"
]

ANCHOR_REGEX = re.compile("|".join(CLINICAL_ANCHORS), re.IGNORECASE)

def main():
    with open(INPUT_D2, "r", encoding="utf-8") as f:
        vignettes = [json.loads(l) for l in f if json.loads(l).get("source") == "medqa"]

    stripped_records = []
    anchors_removed_count = 0

    for item in vignettes:
        q = item["question"]
        matches = ANCHOR_REGEX.findall(q)

        if matches:
            # Replace findings with neutral non-specific wording
            cleaned_q = ANCHOR_REGEX.sub("unremarkable clinical finding", q)
            entry = json.loads(json.dumps(item))
            entry["question"] = cleaned_q
            entry["perturbation"] = "P5_anchor_finding_removal"
            entry["removed_anchors"] = matches
            stripped_records.append(entry)
            anchors_removed_count += len(matches)

    print(f"Processed {len(vignettes)} MedQA test items.")
    print(f"Items with confirmed clinical anchors removed: {len(stripped_records)}")
    print(f"Total clinical anchors stripped: {anchors_removed_count}")

    with open(OUTPUT_P5, "w", encoding="utf-8") as f:
        for r in stripped_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    h = hashlib.sha256(open(OUTPUT_P5, "rb").read()).hexdigest()
    print(f"P5 SHA-256: {h}")

if __name__ == "__main__":
    main()