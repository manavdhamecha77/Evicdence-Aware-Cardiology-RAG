import os
import re
import json
import hashlib
from pypdf import PdfReader

GUIDELINES_DIR = os.path.join("Knowledge_Base", "raw_guidelines")
OUTPUT_CHUNKS_DIR = os.path.join("Knowledge_Base", "indexed_chunks")
OUTPUT_FILE = os.path.join(OUTPUT_CHUNKS_DIR, "D4_cardiology_kb.jsonl")
MANIFEST_FILE = "D4_manifest.txt"

# Chunking configuration (word approximations of ~512 tokens and ~64 overlap)
CHUNK_SIZE_WORDS = 380
OVERLAP_WORDS = 50

def clean_page_text(text: str) -> str:
    """Removes standard PDF artifacts, page headers/footers, and irregular whitespace."""
    if not text:
        return ""
    # Strip common repetitive journal header/footer patterns
    text = re.sub(r"European Heart Journal\s*\(\d{4}\).*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Circulation\s*\.\s*\d{4};.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Downloaded from http.*", "", text)
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    
    # Replace non-breaking spaces and line wraps inside sentences
    text = text.replace("\xa0", " ")
    text = re.sub(r"(?<=[a-zA-Z])-\n(?=[a-zA-Z])", "", text)  # join hyphenated words
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sliding_window_chunk(words, chunk_size, overlap):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:  # skip trailing fragments
            continue
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def main():
    if not os.path.exists(OUTPUT_CHUNKS_DIR):
        os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} guideline PDFs in {GUIDELINES_DIR}")
    if not pdf_files:
        print("ERROR: No PDF files located in raw_guidelines folder. Please copy them over.")
        return

    all_chunks = []
    total_pages_processed = 0

    for pdf_name in pdf_files:
        pdf_path = os.path.join(GUIDELINES_DIR, pdf_name)
        print(f"Processing: {pdf_name} ...")
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        total_pages_processed += num_pages

        doc_words = []
        page_word_map = []  # tracks word index to source page

        for page_idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = clean_page_text(raw_text)
            words = cleaned.split()
            start_pos = len(doc_words)
            doc_words.extend(words)
            end_pos = len(doc_words)
            page_word_map.append((start_pos, end_pos, page_idx))

        # Perform chunking across continuous document text
        doc_chunks = sliding_window_chunk(doc_words, CHUNK_SIZE_WORDS, OVERLAP_WORDS)
        print(f" -> Extracted {len(doc_chunks)} chunks across {num_pages} pages.")

        for c_idx, chunk_str in enumerate(doc_chunks):
            # Estimate primary pages covered
            approx_word_idx = c_idx * (CHUNK_SIZE_WORDS - OVERLAP_WORDS)
            covering_page = 1
            for start_p, end_p, pg_num in page_word_map:
                if start_p <= approx_word_idx < end_p:
                    covering_page = pg_num
                    break

            chunk_id = f"KB_{os.path.splitext(pdf_name)[0][:18]}_{c_idx:04d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "source_doc": pdf_name,
                "estimated_page": covering_page,
                "token_approx": len(chunk_str.split()),
                "text": chunk_str
            })

    print(f"\nTotal structured chunks produced: {len(all_chunks)} across {total_pages_processed} pages.")

    # Save to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Compute SHA-256 Checksum
    hasher = hashlib.sha256()
    with open(OUTPUT_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256 = hasher.hexdigest()

    print(f"[FROZEN] Saved D4 Knowledge Base chunks to: {OUTPUT_FILE}")
    print(f"SHA-256 Checksum: {sha256}")

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        f.write("D4 KNOWLEDGE BASE FREEZE MANIFEST\n" + "=" * 50 + "\n")
        f.write(f"Source PDFs Processed: {len(pdf_files)}\n")
        f.write(f"Total Pages: {total_pages_processed}\n")
        f.write(f"Total Chunks: {len(all_chunks)}\n")
        f.write(f"Output File: {OUTPUT_FILE}\n")
        f.write(f"SHA-256: {sha256}\n")

    print(f"Manifest written to {MANIFEST_FILE}. D4 is complete.")

if __name__ == "__main__":
    main()