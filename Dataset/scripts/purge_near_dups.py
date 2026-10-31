import json
import hashlib

DROP_IDS = {
    "medmcqa_e5279af1-83ea-41be-b8fd-ac69cb208744",
    "medmcqa_efd050d5-d475-4c0c-afbc-4053000e3387",
    "medmcqa_31092a90-32e7-484c-8086-86f21abc2ac5",
    "medmcqa_977d2d60-35bd-477a-8a77-e302d1ced881"
}

def clean_file(path, drop_set, id_key="id"):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    kept = [r for r in rows if r.get(id_key) not in drop_set]
    with open(path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return len(kept), h

# 1. Prune D2 Test
test_len, test_hash = clean_file("artifacts/D2_test_clean.jsonl", DROP_IDS)
print(f"Pruned D2 test: {test_len} items remaining | SHA-256: {test_hash}")

# 2. Prune D3 P1 (1:1 test mirror)
p1_len, p1_hash = clean_file("artifacts/D3_P1_permutations.jsonl", DROP_IDS)
print(f"Pruned D3 P1: {p1_len} items remaining | SHA-256: {p1_hash}")

# 3. Check and prune from P3, P4, P5 if present
for p_name in ["D3_P3_noise.jsonl", "D3_P4_jitter.jsonl", "D3_P5_stripped.jsonl"]:
    p_len, p_hash = clean_file(f"artifacts/{p_name}", DROP_IDS)
    print(f"Checked {p_name}: {p_len} items | SHA-256: {p_hash}")

# 4. Read other unchanged D2 hashes for manifest rewrite
val_hash = hashlib.sha256(open("artifacts/D2_val_clean.jsonl", "rb").read()).hexdigest()
train_hash = hashlib.sha256(open("artifacts/D2_train_clean.jsonl", "rb").read()).hexdigest()

with open("manifests/D2_manifest.txt", "w", encoding="utf-8") as f:
    f.write("D2 DATASET FREEZE MANIFEST\n" + "=" * 60 + "\n")
    f.write(f"File: artifacts/D2_test_clean.jsonl\nCount: {test_len}\nSHA-256: {test_hash}\n\n")
    f.write(f"File: artifacts/D2_val_clean.jsonl\nCount: 613\nSHA-256: {val_hash}\n\n")
    f.write(f"File: artifacts/D2_train_clean.jsonl\nCount: 4910\nSHA-256: {train_hash}\n\n")

# 5. Read hashes for D3 manifest rewrite
p3_hash = hashlib.sha256(open("artifacts/D3_P3_noise.jsonl", "rb").read()).hexdigest()
p4_hash = hashlib.sha256(open("artifacts/D3_P4_jitter.jsonl", "rb").read()).hexdigest()
p5_hash = hashlib.sha256(open("artifacts/D3_P5_stripped.jsonl", "rb").read()).hexdigest()

with open("manifests/D3_manifest.txt", "w", encoding="utf-8") as f:
    f.write("D3 PERTURBATION TEST SETS MANIFEST\n" + "=" * 50 + "\n")
    f.write(f"File: artifacts/D3_P1_permutations.jsonl\nCount: {p1_len}\nSHA-256: {p1_hash}\n\n")
    f.write(f"File: artifacts/D3_P3_noise.jsonl\nCount: 273\nSHA-256: {p3_hash}\n\n")
    f.write(f"File: artifacts/D3_P4_jitter.jsonl\nCount: 200\nSHA-256: {p4_hash}\n\n")
    f.write(f"File: artifacts/D3_P5_stripped.jsonl\nCount: 21\nSHA-256: {p5_hash}\n\n")

print("\nUpdated D2_manifest.txt and D3_manifest.txt successfully.")