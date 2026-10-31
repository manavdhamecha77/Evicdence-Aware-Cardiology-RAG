import json
import re
import requests
import xml.etree.ElementTree as ET
import hashlib

INPUT_RAW = "D1_case_reports_raw.jsonl"
OUTPUT_STAGED = "D1_cases_staged.jsonl"

NON_HUMAN_KEYWORDS = [
    r"\bcat\b", r"\bfeline\b", r"\bdog\b", r"\bcanine\b", 
    r"\bhorse\b", r"\bequine\b", r"\bswine\b", r"\bporcine\b", 
    r"\bmurine\b", r"\brat\b", r"\bmouse\b"
]

def is_veterinary(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in NON_HUMAN_KEYWORDS)

def clean_xml_text(element) -> str:
    """Recursively extract plain text from XML element."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()

def parse_full_text(pmcid: str) -> dict:
    """Fetch full Europe PMC XML and parse distinct clinical sections."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        
        root = ET.fromstring(resp.content)
        sections = {"hpi": "", "exam": "", "diagnostics": "", "outcome": ""}
        
        # Search all body sections
        for sec in root.iter("sec"):
            title_elem = sec.find("title")
            title = clean_xml_text(title_elem).lower() if title_elem is not None else ""
            body_text = " ".join(clean_xml_text(p) for p in sec.findall("p"))
            
            if any(k in title for k in ["presentation", "history", "case description", "case presentation", "case report", "summary"]):
                if not sections["hpi"]:
                    sections["hpi"] = body_text
            elif any(k in title for k in ["investigation", "diagnostic", "examination", "imaging", "laboratory", "workup"]):
                sections["diagnostics"] += " " + body_text
            elif any(k in title for k in ["treatment", "outcome", "management", "follow-up", "discussion"]):
                sections["outcome"] += " " + body_text

        return sections
    except Exception:
        return {}

def extract_stages_from_text(abstract: str, full_sections: dict) -> dict:
    """
    Constructs 4-stage progressive reveal schema for D1 & D6.
    Falls back to structured abstract extraction if XML sections are missing.
    """
    hpi = full_sections.get("hpi", "")
    diagnostics = full_sections.get("diagnostics", "")
    outcome = full_sections.get("outcome", "")
    
    # Fallback to abstract sentences if full-text XML is sparse
    if len(hpi) < 100:
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        hpi = " ".join(sentences[:2]) if len(sentences) >= 2 else abstract
        diagnostics = " ".join(sentences[2:4]) if len(sentences) >= 4 else ""
        outcome = " ".join(sentences[4:]) if len(sentences) >= 5 else ""

    return {
        "stage_1_presentation": hpi.strip()[:1200],
        "stage_2_diagnostics": diagnostics.strip()[:1500] if diagnostics else "Initial diagnostics pending.",
        "stage_3_outcome_and_intervention": outcome.strip()[:1200] if outcome else "Intervention and follow-up pending."
    }

def main():
    print(f"Reading raw cases from {INPUT_RAW}...")
    with open(INPUT_RAW, "r", encoding="utf-8") as f:
        raw_cases = [json.loads(line) for line in f]

    staged_cases = []
    dropped_vet = 0

    print(f"Loaded {len(raw_cases)} raw cases. Filtering and structuring into 4 clinical stages...")

    for case in raw_cases:
        title = case.get("title", "")
        abstract = case.get("abstract", "")
        pmcid = case.get("case_id", "")

        # 1. Filter out veterinary/animal studies
        if is_veterinary(title) or is_veterinary(abstract):
            dropped_vet += 1
            continue

        # 2. Fetch and parse clinical sections
        full_sections = parse_full_text(pmcid)
        stages = extract_stages_from_text(abstract, full_sections)

        staged_entry = {
            "case_id": pmcid,
            "doi": case.get("doi", ""),
            "title": title,
            "year": case.get("year", ""),
            "presentation_stage_1": stages["stage_1_presentation"],
            "diagnostics_stage_2": stages["stage_2_diagnostics"],
            "intervention_stage_3": stages["stage_3_outcome_and_intervention"],
            "ground_truth_abstract": abstract
        }
        staged_cases.append(staged_entry)

    print(f"Dropped {dropped_vet} veterinary cases.")
    print(f"Total human clinical cases staged: {len(staged_cases)}")

    with open(OUTPUT_STAGED, "w", encoding="utf-8") as f:
        for item in staged_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Cryptographic freeze hash
    hasher = hashlib.sha256()
    with open(OUTPUT_STAGED, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()

    print(f"\n[FROZEN] Saved staged cases to {OUTPUT_STAGED}")
    print(f"SHA-256 Checksum: {sha256}")

    with open("D1_manifest.txt", "w", encoding="utf-8") as f:
        f.write(f"D1 REAL-WORLD CLINICAL CASES MANIFEST\n{'='*50}\n")
        f.write(f"File: {OUTPUT_STAGED}\nCount: {len(staged_cases)}\nSHA-256: {sha256}\n")

if __name__ == "__main__":
    main()