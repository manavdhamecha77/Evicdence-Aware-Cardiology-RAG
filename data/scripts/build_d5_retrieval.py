import json
import re
import hashlib
from collections import defaultdict

D4_CHUNKS_FILE = r"knowledge base\indexed chunks\D4_cardiology_kb.jsonl"
D2_TEST_FILE = "D2_test_clean.jsonl"
OUTPUT_D5_FILE = "D5_retrieval_benchmark.jsonl"
MANIFEST_FILE = "D5_manifest.txt"

# Core clinical domain anchor mapping
DOMAIN_KEYWORDS = {
    "Aortic Stenosis": ["aortic stenosis", "transcatheter aortic valve", "tavr", "savr", "valve area"],
    "Mitral Regurgitation": ["mitral regurgitation", "teer", "mitral valve repair", "systolic anterior motion"],
    "Acute Coronary Syndrome": ["nstemi", "stemi", "culprit lesion", "percutaneous coronary intervention", "dual antiplatelet"],
    "Heart Failure": ["heart failure", "sglt2", "arni", "sacubitril", "reduced ejection fraction", "hfpef"],
    "Infective Endocarditis": ["infective endocarditis", "duke criteria", "blood culture", "vegetation"]
}

def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())

def main():
    print(f"Loading D4 guideline chunks from {D4_CHUNKS_FILE}...")
    with open(D4_CHUNKS_FILE, "r", encoding="utf-8") as f:
        kb_chunks = [json.loads(line) for line in f]
    print(f"Loaded {len(kb_chunks)} chunks.")

    print(f"Loading D2 test items from {D2_TEST_FILE}...")
    with open(D2_TEST_FILE, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f]

    # Pre-index chunks by domain keywords
    chunk_scores = defaultdict(dict)
    for chunk in kb_chunks:
        chunk_text_clean = normalize_text(chunk["text"])
        for domain, terms in DOMAIN_KEYWORDS.items():
            matches = sum(1 for term in terms if term in chunk_text_clean)
            if matches >= 2:
                chunk_scores[domain][chunk["chunk_id"]] = matches

    d5_pairs = []
    for item in test_items:
        q_clean = normalize_text(item["question"])
        matched_domain = None
        for domain, terms in DOMAIN_KEYWORDS.items():
            if any(term in q_clean for term in terms):
                matched_domain = domain
                break

        if matched_domain and matched_domain in chunk_scores:
            # Top 3 most relevant guideline chunks as ground truth
            top_relevant = sorted(chunk_scores[matched_domain].items(), key=lambda x: x[1], reverse=True)[:3]
            target_ids = [t[0] for t in top_relevant]
            if target_ids:
                d5_pairs.append({
                    "query_id": item["id"],
                    "domain": matched_domain,
                    "query": item["question"],
                    "positive_chunk_ids": target_ids
                })

    print(f"Extracted {len(d5_pairs)} query-retrieval ground truth pairs.")

    with open(OUTPUT_D5_FILE, "w", encoding="utf-8") as f:
        for pair in d5_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    hasher = hashlib.sha256()
    with open(OUTPUT_D5_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()

    print(f"[FROZEN] Saved D5 Retrieval Pairs to {OUTPUT_D5_FILE}")
    print(f"SHA-256: {sha256}")

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write("D5 RETRIEVAL BENCHMARK MANIFEST\n" + "=" * 50 + "\n")
        f.write(f"Total Evaluated Retrieval Pairs: {len(d5_pairs)}\n")
        f.write(f"File: {OUTPUT_D5_FILE}\n")
        f.write(f"SHA-256: {sha256}\n")

if __name__ == "__main__":
    main()
