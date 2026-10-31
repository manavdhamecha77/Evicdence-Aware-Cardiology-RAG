import json
import random
import re
import hashlib

random.seed(42)

INPUT_D2_TEST = "D2_test_clean.jsonl"

NOISE_ADDITIONS = [
    " The patient also notes a history of mild seasonal allergic rhinitis, which is well managed.",
    " The patient reports a remote history of a resolved sprained ankle 3 years ago.",
    " Mild bilateral bunions are noted on lower extremity physical examination.",
    " The patient mentions using over-the-counter artificial tears occasionally for mild dry eyes."
]

BUZZWORDS = [
    r"boot-shaped heart",
    r"egg-shaped heart",
    r"biconvex spindle-shaped vacuoles",
    r"spindle-shaped vacuoles",
    r"electrical alternans",
    r"Aschoff bodies",
    r"Anitschkow cells",
    r"water-hammer pulse",
    r"pulsus paradoxus"
]

def perturb_p1_permute(item: dict) -> dict:
    """P1: Permutes option order and updates ground truth key."""
    new_item = json.loads(json.dumps(item))
    options = item["options"]
    correct_text = options[item["answer"]]
    
    pairs = list(options.items())
    keys = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    random.shuffle(values)
    
    new_options = {keys[i]: values[i] for i in range(len(keys))}
    new_answer = next(k for k, v in new_options.items() if v == correct_text)
    
    new_item["options"] = new_options
    new_item["answer"] = new_answer
    new_item["perturbation"] = "P1_permutation"
    return new_item

def perturb_p3_noise(item: dict) -> dict:
    """P3: Appends irrelevant non-contributory clinical distractors."""
    new_item = json.loads(json.dumps(item))
    noise = random.choice(NOISE_ADDITIONS)
    new_item["question"] = item["question"].rstrip() + noise
    new_item["perturbation"] = "P3_noise_injection"
    return new_item

def perturb_p4_jitter(item: dict) -> dict:
    """P4: Applies physiological jitter to non-critical numeric vitals."""
    new_item = json.loads(json.dumps(item))
    q = new_item["question"]

    # Jitter Blood Pressure (e.g., 120/80 -> 124/82)
    def bp_repl(match):
        sys = int(match.group(1)) + random.choice([-4, -2, 2, 4])
        dia = int(match.group(2)) + random.choice([-3, -1, 1, 3])
        return f"{sys}/{dia}"

    q = re.sub(r"\b(\d{2,3})/(\d{2,3})\b", bp_repl, q)

    # Jitter Pulse / Heart Rate (e.g., 72/min -> 75/min)
    def hr_repl(match):
        val = int(match.group(1)) + random.choice([-3, -2, 2, 3])
        return f"{val}{match.group(2)}"

    q = re.sub(r"\b(\d{2,3})(\s*(?:beats/min|bpm|/min))", hr_repl, q)

    new_item["question"] = q
    new_item["perturbation"] = "P4_numerical_jitter"
    return new_item

def perturb_p5_strip(item: dict) -> dict:
    """P5: Strips high-yield pathognomonic anchor phrases."""
    new_item = json.loads(json.dumps(item))
    q = new_item["question"]
    for bw in BUZZWORDS:
        q = re.sub(bw, "unremarkable diagnostic finding", q, flags=re.IGNORECASE)
    new_item["question"] = q
    new_item["perturbation"] = "P5_finding_removal"
    return new_item

def main():
    print(f"Loading base test set: {INPUT_D2_TEST}")
    with open(INPUT_D2_TEST, "r", encoding="utf-8") as f:
        base_test = [json.loads(line) for line in f]

    # Target long MedQA clinical vignettes for P3, P4, P5
    medqa_vignettes = [x for x in base_test if x.get("source") == "medqa"]
    print(f"Total base items: {len(base_test)} (MedQA vignettes: {len(medqa_vignettes)})")

    p1_set = [perturb_p1_permute(x) for x in base_test]
    p3_set = [perturb_p3_noise(x) for x in medqa_vignettes]
    p4_set = [perturb_p4_jitter(x) for x in medqa_vignettes]
    p5_set = [perturb_p5_strip(x) for x in medqa_vignettes]

    perturbation_files = {
        "D3_P1_permutations.jsonl": p1_set,
        "D3_P3_noise.jsonl": p3_set,
        "D3_P4_jitter.jsonl": p4_set,
        "D3_P5_stripped.jsonl": p5_set,
    }

    manifest_lines = ["D3 PERTURBATION TEST SETS MANIFEST\n" + "=" * 50 + "\n"]

    for filename, dataset in perturbation_files.items():
        with open(filename, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        hasher = hashlib.sha256()
        with open(filename, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        sha256 = hasher.hexdigest()
        
        print(f"Generated {filename:26s} | Count: {len(dataset):4d} | SHA-256: {sha256}")
        manifest_lines.append(f"File: {filename}\nCount: {len(dataset)}\nSHA-256: {sha256}\n\n")

    with open("D3_manifest.txt", "w", encoding="utf-8") as f:
        f.writelines(manifest_lines)

    print("\nManifest written to D3_manifest.txt. D3 is complete.")

if __name__ == "__main__":
    main()