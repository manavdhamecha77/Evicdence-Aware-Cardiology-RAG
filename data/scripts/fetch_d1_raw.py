import requests
import json
import time
import xml.etree.ElementTree as ET

# -------------------------------------------------------------------
# Configuration: Target 120-150 Open-Access Full-Text Cardiology Cases
# -------------------------------------------------------------------
BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
QUERY = '(TITLE:"case report" OR ABSTRACT:"case report") AND (KW:"cardiology" OR KW:"myocardial infarction" OR KW:"heart failure" OR KW:"cardiomyopathy" OR KW:"arrhythmia" OR KW:"endocarditis") AND OPEN_ACCESS:y'

params = {
    "query": QUERY,
    "format": "json",
    "pageSize": 100,
    "resultType": "core"
}

output_file = "D1_case_reports_raw.jsonl"
cases_collected = []

print("Fetching case report metadata from Europe PMC...")
cursor_mark = "*"

while len(cases_collected) < 120:
    params["cursorMark"] = cursor_mark
    resp = requests.get(BASE_URL, params=params)
    if resp.status_code != 200:
        print(f"Error querying API: {resp.status_code}")
        break
        
    data = resp.json()
    results = data.get("resultList", {}).get("result", [])
    if not results:
        break

    for item in results:
        pmcid = item.get("pmcid")
        title = item.get("title", "")
        abstract = item.get("abstractText", "")
        
        # Only retain records with accessible PMCID and full text
        if pmcid and abstract:
            cases_collected.append({
                "case_id": pmcid,
                "pmid": item.get("pmid", ""),
                "doi": item.get("doi", ""),
                "title": title,
                "journal": item.get("journalTitle", ""),
                "year": item.get("pubYear", ""),
                "abstract": abstract,
                # Europe PMC full text XML endpoint
                "full_text_xml_url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            })
            if len(cases_collected) >= 120:
                break

    cursor_mark = data.get("nextCursorMark")
    if not cursor_mark:
        break
    time.sleep(0.5)

print(f"Retrieved {len(cases_collected)} cardiology case reports.")

with open(output_file, "w", encoding="utf-8") as f:
    for case in cases_collected:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")

print(f"Saved to {output_file}")