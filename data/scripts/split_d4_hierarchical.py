import os
import re
import json
import hashlib
from pypdf import PdfReader

GUIDELINES_DIR = os.path.join("knowledge base", "raw guidelines")
OUTPUT_CHUNKS_DIR = os.path.join("knowledge base", "indexed chunks")
OUTPUT_FILE = os.path.join(OUTPUT_CHUNKS_DIR, "D4_cardiology_kb_hierarchical.jsonl")
MANIFEST_FILE = os.path.join("manifests", "D4_manifest.txt")

HEADING_REGEX = re.compile(
    r"^(?:\d+\.|\d+\.\d+|\b[IVXLCDM]+\.|\bSECTION\s+\d+)\s+([A-Z][A-Za-z0-9\s,\-–:]{3,80})$",
    re.M
)

MAX_WORDS = 500
OVERLAP_WORDS = 50

def clean_text(t: str) -> str:
    t = re.sub(r"European Heart Journal\s*\(\d{4}\).*", "", t, flags=re.I)
    t = re.sub(r"Circulation\s*\.\s*\d{4};.*", "", t, flags=re.I)
    t = re.sub(r"Downloaded from http.*", "", t)
    t = t.replace("\xa0", " ")
    return re.sub(r"\s+", " ", t).strip()

def chunk_section_words(words, max_w, overlap):
    if len(words) <= max_w:
        return [" ".join(words)]
    chunks = []
    step = max_w - overlap
    for i in range(0, len(words), step):
        slice_words = words[i:i + max_w]
        if len(slice_words) >= 40:
            chunks.append(" ".join(slice_words))
        if i + max_w >= len(words):
            break
    return chunks

def main():
    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    chunks = []

    for pdf_name in pdf_files:
        pdf_path = os.path.join(GUIDELINES_DIR, pdf_name)
        print(f"Parsing and capping hierarchical sections: {pdf_name}")
        reader = PdfReader(pdf_path)
        
        current_heading = "General Guideline Principles"
        current_text = []

        def flush_section(heading, text_list, page_num):
            sec_words = " ".join(text_list).split()
            if len(sec_words) < 50:
                return
            sub_chunks = chunk_section_words(sec_words, MAX_WORDS, OVERLAP_WORDS)
            for part_idx, c_text in enumerate(sub_chunks):
                part_label = f" (Part {part_idx + 1})" if len(sub_chunks) > 1 else ""
                chunk_id = f"KB_HIER_{os.path.splitext(pdf_name)[0][:14]}_{len(chunks):04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source_doc": pdf_name,
                    "section_heading": f"{heading}{part_label}",
                    "page": page_num,
                    "token_approx": len(c_text.split()),
                    "text": c_text
                })

        for p_idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            for line in raw.split("\n"):
                l_strip = line.strip()
                h_match = HEADING_REGEX.match(l_strip)
                if h_match:
                    flush_section(current_heading, current_text, p_idx)
                    current_heading = h_match.group(0).strip()
                    current_text = []
                else:
                    cl = clean_text(l_strip)
                    if cl:
                        current_text.append(cl)

        flush_section(current_heading, current_text, len(reader.pages))

    print(f"Total capped heading-aware chunks: {len(chunks)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    h_hier = hashlib.sha256(open(OUTPUT_FILE, "rb").read()).hexdigest()
    print(f"Hierarchical Arm SHA-256: {h_hier}")

    # Re-read naive arm hash to write unified D4 manifest
    naive_file = os.path.join(OUTPUT_CHUNKS_DIR, "D4_cardiology_kb.jsonl")
    h_naive = hashlib.sha256(open(naive_file, "rb").read()).hexdigest()

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write("D4 KNOWLEDGE BASE FREEZE MANIFEST\n" + "=" * 50 + "\n")
        f.write(f"Source PDFs Processed: {len(pdf_files)}\n\n")
        f.write("Arm 1 (Naive Fixed-Size Chunks):\n")
        f.write(f"File: {naive_file}\nCount: 959\nSHA-256: {h_naive}\n\n")
        f.write("Arm 2 (Heading-Aware Capped Chunks):\n")
        f.write(f"File: {OUTPUT_FILE}\nCount: {len(chunks)}\nSHA-256: {h_hier}\n")

    print(f"Updated {MANIFEST_FILE}")

if __name__ == "__main__":
    main()
