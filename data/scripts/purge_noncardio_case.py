import json
import hashlib

D1_PATH = "artifacts/D1_cases_staged.jsonl"
D6_PATH = "artifacts/D6_progressive_reveal.jsonl"
D1_MANIFEST = "manifests/D1_manifest.txt"
D6_MANIFEST = "manifests/D6_manifest.txt"

# 1. Purge Row 34 from D1
with open(D1_PATH, "r", encoding="utf-8") as f:
    d1_cases = [json.loads(line) for line in f]

target_case = d1_cases[34]
target_doi = target_case.get("doi", "")
target_pmcid = target_case.get("pmcid", "")
print(f"Purging case at index 34: Title='{target_case.get('title')}' | DOI={target_doi}")

d1_clean = [c for idx, c in enumerate(d1_cases) if idx != 34]

with open(D1_PATH, "w", encoding="utf-8") as f:
    for c in d1_clean:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

d1_hash = hashlib.sha256(open(D1_PATH, "rb").read()).hexdigest()
with open(D1_MANIFEST, "w", encoding="utf-8") as f:
    f.write("D1 REAL-WORLD CLINICAL CASES MANIFEST\n" + "=" * 50 + "\n")
    f.write(f"File: {D1_PATH}\nCount: {len(d1_clean)}\nSHA-256: {d1_hash}\n")

print(f"D1 Updated: {len(d1_clean)} cases (Count=101) | SHA-256: {d1_hash}")

# 2. Purge matching trajectory from D6
with open(D6_PATH, "r", encoding="utf-8") as f:
    d6_trajectories = [json.loads(line) for line in f]

d6_clean = [
    t for t in d6_trajectories 
    if t.get("doi") != target_doi and target_pmcid not in t.get("trajectory_id", "")
]

with open(D6_PATH, "w", encoding="utf-8") as f:
    for t in d6_clean:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

d6_hash = hashlib.sha256(open(D6_PATH, "rb").read()).hexdigest()
with open(D6_MANIFEST, "w", encoding="utf-8") as f:
    f.write("D6 PROGRESSIVE REVEAL TRAJECTORY MANIFEST\n" + "=" * 50 + "\n")
    f.write(f"File: {D6_PATH}\nCount: {len(d6_clean)}\nSHA-256: {d6_hash}\n")

print(f"D6 Updated: {len(d6_clean)} trajectories (Count=101) | SHA-256: {d6_hash}")