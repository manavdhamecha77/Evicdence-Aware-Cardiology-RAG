import json
import hashlib

DOI_VAL = "10.4314/ajhs.v38i2.11"
D1_PATH = "artifacts/D1_cases_staged.jsonl"
D6_PATH = "artifacts/D6_progressive_reveal.jsonl"

# 1. Update D1
with open(D1_PATH, "r", encoding="utf-8") as f:
    d1_cases = [json.loads(line) for line in f]

for c in d1_cases:
    if "13399423" in str(c.get("pmcid", "")) or "13399423" in str(c.get("id", "")):
        c["doi"] = DOI_VAL

with open(D1_PATH, "w", encoding="utf-8") as f:
    for c in d1_cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

d1_hash = hashlib.sha256(open(D1_PATH, "rb").read()).hexdigest()
with open("manifests/D1_manifest.txt", "w", encoding="utf-8") as f:
    f.write("D1 REAL-WORLD CLINICAL CASES MANIFEST\n" + "=" * 50 + "\n")
    f.write(f"File: {D1_PATH}\nCount: {len(d1_cases)}\nSHA-256: {d1_hash}\n")

# 2. Update D6
with open(D6_PATH, "r", encoding="utf-8") as f:
    d6_trajs = [json.loads(line) for line in f]

for t in d6_trajs:
    if "13399423" in t.get("trajectory_id", ""):
        t["doi"] = DOI_VAL

with open(D6_PATH, "w", encoding="utf-8") as f:
    for t in d6_trajs:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

d6_hash = hashlib.sha256(open(D6_PATH, "rb").read()).hexdigest()
with open("manifests/D6_manifest.txt", "w", encoding="utf-8") as f:
    f.write("D6 PROGRESSIVE REVEAL TRAJECTORY MANIFEST\n" + "=" * 50 + "\n")
    f.write(f"File: {D6_PATH}\nCount: {len(d6_trajs)}\nSHA-256: {d6_hash}\n")

print(f"Patched DOI {DOI_VAL} for PMC13399423 across D1 & D6.")
print(f"D1 SHA-256: {d1_hash}")
print(f"D6 SHA-256: {d6_hash}")