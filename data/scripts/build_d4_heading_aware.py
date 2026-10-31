import os, re, json, hashlib
from pypdf import PdfReader

GUIDELINES_DIR = os.path.join("knowledge base", "raw guidelines")
OUTPUT_CHUNKS_DIR = os.path.join("knowledge base", "indexed chunks")
OUTPUT_FILE = os.path.join(OUTPUT_CHUNKS_DIR, "D4_cardiology_kb_hierarchical.jsonl")

# Section heading detection pattern for ESC & ACC/AHA guidelines
HEADING_REGEX = re.compile(r"^(?:\d+\.|\d+\.\d+|\b[IVXLCDM]+\.|\bSECTION\s+\d+)\s+([A-Z][A-Za-z0-9\s,\-–:]{3,80})$", re.M)

def clean_text(t: str) -> str:
    t = re.sub(r"European Heart Journal\s*\(\d{4}\).*", "", t, flags=re.I)
    t = re.sub(r"Circulation\s*\.\s*\d{4};.*", "", t, flags=re.I)
    t = re.sub(r"Downloaded from http.*", "", t)
    t = t.replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()

def main():
    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    chunks = []

    for pdf_name in pdf_files:
        pdf_path = os.path.join(GUIDELINES_DIR, pdf_name)
        print(f"Parsing hierarchical sections from: {pdf_name}")
        reader = PdfReader(pdf_path)
        
        current_heading = "General Guideline Principles"
        current_text = []

        for p_idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            lines = raw.split("\n")
            
            for line in lines:
                l_strip = line.strip()
                h_match = HEADING_REGEX.match(l_strip)
                if h_match:
                    # Flush accumulated section text as a structured chunk
                    sec_words = " ".join(current_text).split()
                    if len(sec_words) >= 60:
                        chunk_id = f"KB_HIER_{os.path.splitext(pdf_name)[0][:14]}_{len(chunks):04d}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "source_doc": pdf_name,
                            "section_heading": current_heading,
                            "page": p_idx,
                            "token_approx": len(sec_words),
                            "text": " ".join(sec_words)
                        })
                    current_heading = h_match.group(0).strip()
                    current_text = []
                else:
                    cleaned_line = clean_text(l_strip)
                    if cleaned_line:
                        current_text.append(cleaned_line)

        # Flush final section
        if current_text:
            sec_words = " ".join(current_text).split()
            if len(sec_words) >= 60:
                chunks.append({
                    "chunk_id": f"KB_HIER_{os.path.splitext(pdf_name)[0][:14]}_{len(chunks):04d}",
                    "source_doc": pdf_name,
                    "section_heading": current_heading,
                    "page": len(reader.pages),
                    "token_approx": len(sec_words),
                    "text": " ".join(sec_words)
                })

    print(f"Total heading-aware chunks produced: {len(chunks)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    h = hashlib.sha256(open(OUTPUT_FILE, "rb").read()).hexdigest()
    print(f"D4 Heading-Aware Checksum: {h}")

if __name__ == "__main__":
    main()
