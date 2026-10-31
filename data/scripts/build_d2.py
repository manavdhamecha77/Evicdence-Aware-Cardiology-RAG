import json
import re
import hashlib
from collections import defaultdict

# ---------------------------------------------------------
# 1. OCR Typo Repair Dictionary
# ---------------------------------------------------------
OCR_REPLACEMENTS = {
    r"\bhea\b": "heart",
    r"\bHea\b": "Heart",
    r"\bAoic\b": "Aortic",
    r"\baoic\b": "aortic",
    r"\baery\b": "artery",
    r"\baeries\b": "arteries",
    r"\bshoness\b": "shortness",
    r"\bsho\b": "short",
    r"\brepo\b": "report",
    r"\bcaheterization\b": "catheterization",
    r"\bcardioveer\b": "cardioverter",
    r"\bmateranl\b": "maternal",
    r"\bble\b": "viable",
}

def clean_ocr(text: str) -> str:
    if not text:
        return ""
    for pattern, replacement in OCR_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)
    return text

def canonical_key(text: str) -> str:
    """Standardizes string to identify verbatim and near-duplicate stems."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

# ---------------------------------------------------------
# 2. Main Cleaning & Leakage-Free Splitting
# ---------------------------------------------------------
def main():
    input_file = "cardiology_cleaned_dataset.jsonl"
    print(f"Loading raw records from: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Total raw items loaded: {len(records)}")

    # Step A: Apply OCR cleaning across questions, options, and explanations
    for item in records:
        item["question"] = clean_ocr(item["question"])
        item["explanation"] = clean_ocr(item.get("explanation", ""))
        for opt_k in item["options"]:
            item["options"][opt_k] = clean_ocr(item["options"][opt_k])

    # Step B: Group by canonical question key to detect internal conflicts and duplicates
    grouped = defaultdict(list)
    for item in records:
        ckey = canonical_key(item["question"])
        grouped[ckey].append(item)

    clean_deduped_records = []
    dropped_conflicts = 0
    internal_duplicates_collapsed = 0

    for ckey, items in grouped.items():
        answers = {it["answer"] for it in items}
        if len(answers) > 1:
            # Drop questions with conflicting ground-truth labels (MedMCQA label noise)
            dropped_conflicts += len(items)
            continue
        
        # All instances agree on the label -> Keep 1 representative canonical instance
        if len(items) > 1:
            internal_duplicates_collapsed += (len(items) - 1)
        
        # Prefer the record with the longest explanation or clearest text
        best_item = max(items, key=lambda x: len(x.get("explanation", "")))
        clean_deduped_records.append(best_item)

    print(f"Dropped {dropped_conflicts} items due to conflicting duplicate answers.")
    print(f"Collapsed {internal_duplicates_collapsed} redundant duplicate copies.")
    print(f"Clean, unique dataset size: {len(clean_deduped_records)}")

    # Step C: Deterministic Split with Zero Cross-Split Leakage
    # Separate into MedQA (long vignettes) and MedMCQA (shorter knowledge stems)
    medqa_records = [r for r in clean_deduped_records if r.get("source") == "medqa"]
    medmcqa_records = [r for r in clean_deduped_records if r.get("source") == "medmcqa"]

    print(f"MedQA available: {len(medqa_records)} | MedMCQA available: {len(medmcqa_records)}")

    # Sort deterministically by question text before slicing to ensure reproducibility
    medqa_records.sort(key=lambda x: x["question"])
    medmcqa_records.sort(key=lambda x: x["question"])

    # Target D2 Test Size: ~570-600 (Balanced between MedQA vignettes and MedMCQA items)
    # Using 10% test allocation across both subsets
    test_medqa_count = int(len(medqa_records) * 0.10)
    val_medqa_count = int(len(medqa_records) * 0.10)

    test_medmcqa_count = int(len(medmcqa_records) * 0.10)
    val_medmcqa_count = int(len(medmcqa_records) * 0.10)

    # Slice strictly non-overlapping partitions
    test_set = (
        medqa_records[:test_medqa_count] + 
        medmcqa_records[:test_medmcqa_count]
    )
    val_set = (
        medqa_records[test_medqa_count : test_medqa_count + val_medqa_count] + 
        medmcqa_records[test_medmcqa_count : test_medmcqa_count + val_medmcqa_count]
    )
    train_set = (
        medqa_records[test_medqa_count + val_medqa_count :] + 
        medmcqa_records[test_medmcqa_count + val_medmcqa_count :]
    )

    # Step D: Strict Cross-Split Verification Assertion
    test_keys = {canonical_key(x["question"]) for x in test_set}
    val_keys = {canonical_key(x["question"]) for x in val_set}
    train_keys = {canonical_key(x["question"]) for x in train_set}

    leakage_train_test = test_keys.intersection(train_keys)
    leakage_val_test = test_keys.intersection(val_keys)
    leakage_train_val = train_keys.intersection(val_keys)

    assert len(leakage_train_test) == 0, f"Critical Leakage: {len(leakage_train_test)} items overlap train and test!"
    assert len(leakage_val_test) == 0, f"Critical Leakage: {len(leakage_val_test)} items overlap val and test!"
    assert len(leakage_train_val) == 0, f"Critical Leakage: {len(leakage_train_val)} items overlap train and val!"

    print("\n[VERIFIED] 0% Cross-split contamination verified across all splits.")

    # Step E: Save Final Frozen Files
    splits = {
        "D2_test_clean.jsonl": test_set,
        "D2_val_clean.jsonl": val_set,
        "D2_train_clean.jsonl": train_set
    }

    print("\nWriting clean files and generating SHA-256 hashes:")
    hash_summary = []
    for filename, dataset in splits.items():
        with open(filename, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        sha256_hash = compute_sha256(filename)
        hash_summary.append((filename, len(dataset), sha256_hash))
        print(f" - {filename:20s}: {len(dataset):5d} items | SHA-256: {sha256_hash}")

    # Step F: Write Manifest File
    with open("D2_manifest.txt", "w", encoding="utf-8") as f:
        f.write("D2 DATASET FREEZE MANIFEST\n")
        f.write("=" * 60 + "\n")
        for fn, count, sha in hash_summary:
            f.write(f"File: {fn}\nCount: {count}\nSHA-256: {sha}\n\n")

    print("\nManifest written to D2_manifest.txt. D2 is frozen and ready.")

if __name__ == "__main__":
    main()