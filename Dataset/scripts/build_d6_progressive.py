import json
import hashlib

INPUT_D1 = "artifacts/D1_cases_staged.jsonl"
OUTPUT_D6 = "artifacts/D6_progressive_reveal.jsonl"
MANIFEST_FILE = "manifests/D6_manifest.txt"

def format_progressive_trajectory(case: dict) -> dict:
    case_id = case.get("pmcid") or case.get("id") or case.get("case_id", "")
    hpi = case.get("stage_1_presentation", "")
    diag = case.get("stage_2_diagnostics", "")
    outcome = case.get("stage_3_intervention", "")
    gold = case.get("gold_primary", "")

    return {
        "trajectory_id": f"D6_{case_id}",
        "doi": case.get("doi", ""),
        "gold_diagnosis": gold,
        "step_1_input": f"CLINICAL PRESENTATION:\n{hpi}\n\nTask: Provide the top 3 differential diagnoses and state the most urgent next diagnostic step.",
        "step_2_input": f"CLINICAL PRESENTATION:\n{hpi}\n\nDIAGNOSTIC WORKUP:\n{diag}\n\nTask: Re-evaluate and rank the differential diagnosis based on these findings.",
        "step_3_input": f"CLINICAL PRESENTATION:\n{hpi}\n\nDIAGNOSTIC WORKUP:\n{diag}\n\nDEFINITIVE INTERVENTION & OUTCOME:\n{outcome}\n\nTask: Confirm the definitive primary diagnosis and outline guideline-directed long-term management.",
        "ground_truth_reference": case.get("ground_truth_abstract", "")
    }

def main():
    print(f"Loading staged cases from {INPUT_D1}...")
    with open(INPUT_D1, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    trajectories = [format_progressive_trajectory(c) for c in cases]
    print(f"Generated {len(trajectories)} multi-stage progressive reveal trajectories.")

    with open(OUTPUT_D6, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    hasher = hashlib.sha256()
    with open(OUTPUT_D6, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()

    print(f"[FROZEN] Saved D6 Progressive-Reveal Trajectories to {OUTPUT_D6}")
    print(f"SHA-256: {sha256}")

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write("D6 PROGRESSIVE REVEAL TRAJECTORY MANIFEST\n" + "=" * 50 + "\n")
        f.write(f"File: {OUTPUT_D6}\n")
        f.write(f"Count: {len(trajectories)}\n")
        f.write(f"SHA-256: {sha256}\n")

    print(f"Manifest written to {MANIFEST_FILE}.")

if __name__ == "__main__":
    main()