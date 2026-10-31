import json
import re
import random
import hashlib

random.seed(42)
INPUT_D2 = "artifacts/D2_test_clean.jsonl"
OUTPUT_P4 = "artifacts/D3_P4_jitter.jsonl"

HR_PATTERN = re.compile(r"\b(?:pulse|heart rate|hr)[:\s]+(?:is\s+)?(\d{2,3})\s*(?:/min|bpm|beats/min)\b", re.I)
BP_PATTERN = re.compile(r"\b(?:bp|blood pressure)[:\s]+(?:is\s+)?(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg)?\b", re.I)

def jitter_hr(val: int) -> int:
    delta = random.choice([-4, -2, 2, 4])
    cand = val + delta
    if val < 60:
        return min(cand, 58)
    elif 60 <= val <= 100:
        return max(62, min(cand, 98))
    else:
        return max(104, cand)

def jitter_sbp(val: int) -> int:
    delta = random.choice([-4, -2, 2, 4])
    cand = val + delta
    if val < 90:
        return min(cand, 88)
    elif 90 <= val < 140:
        return max(94, min(cand, 136))
    elif 140 <= val < 180:
        return max(144, min(cand, 176))
    else:
        return max(184, cand)

def jitter_dbp(val: int) -> int:
    delta = random.choice([-3, -1, 1, 3])
    cand = val + delta
    if val < 60:
        return min(cand, 58)
    elif 60 <= val < 90:
        return max(64, min(cand, 86))
    else:
        return max(94, cand)

def main():
    with open(INPUT_D2, "r", encoding="utf-8") as f:
        vignettes = [json.loads(l) for l in f if json.loads(l).get("source") == "medqa"]

    p4_records = []
    noops = 0

    for item in vignettes:
        orig_q = item["question"]
        mod_q = orig_q

        def replace_hr(m):
            v = int(m.group(1))
            nv = jitter_hr(v)
            return m.group(0).replace(str(v), str(nv))

        def replace_bp(m):
            s, d = int(m.group(1)), int(m.group(2))
            ns, nd = jitter_sbp(s), jitter_dbp(d)
            return m.group(0).replace(f"{s}/{d}", f"{ns}/{nd}")

        mod_q = HR_PATTERN.sub(replace_hr, mod_q)
        mod_q = BP_PATTERN.sub(replace_bp, mod_q)

        # Check if modified
        if mod_q == orig_q:
            # Fallback: jitter temperature or respiratory rate if present to eliminate no-ops
            def replace_rr(m):
                rr = int(m.group(1))
                nrr = rr + random.choice([-2, 2])
                return m.group(0).replace(str(rr), str(nrr))
            mod_q = re.sub(r"\b(?:respirations?|rr)[:\s]+(?:are\s+)?(\d{2})\s*(?:/min|breaths/min)\b", replace_rr, mod_q, flags=re.I)

        if mod_q == orig_q:
            noops += 1
            continue

        entry = json.loads(json.dumps(item))
        entry["question"] = mod_q
        entry["perturbation"] = "P4_band_safe_jitter"
        p4_records.append(entry)

    print(f"Total checkable MedQA items: {len(vignettes)}")
    print(f"Valid jittered items saved: {len(p4_records)} (No-ops dropped: {noops})")

    with open(OUTPUT_P4, "w", encoding="utf-8") as f:
        for r in p4_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    h = hashlib.sha256(open(OUTPUT_P4, "rb").read()).hexdigest()
    print(f"P4 SHA-256: {h}")

if __name__ == "__main__":
    main()