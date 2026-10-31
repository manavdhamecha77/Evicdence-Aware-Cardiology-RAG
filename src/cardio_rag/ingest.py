import json
from pathlib import Path

from .chunking import chunk_text
from .schemas import Chunk

REQUIRED_FIELDS = {"document_id", "title", "publisher", "url", "accessed_date", "license_note", "path"}


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read source manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("source manifest must contain a JSON list")
    seen: set[str] = set()
    base = manifest_path.parent
    documents = []
    for position, record in enumerate(raw, start=1):
        if not isinstance(record, dict) or REQUIRED_FIELDS - record.keys():
            raise ValueError(f"manifest record {position} is missing required fields")
        if not all(isinstance(record[key], str) and record[key].strip() for key in REQUIRED_FIELDS):
            raise ValueError(f"manifest record {position} has empty or non-text fields")
        if record["document_id"] in seen:
            raise ValueError(f"duplicate document_id: {record['document_id']}")
        seen.add(record["document_id"])
        documents.append({**record, "_path": str((base / record["path"]).resolve())})
    return documents


def ingest(manifest_path: Path, output_path: Path, chunk_words: int = 550, overlap_words: int = 80) -> int:
    manifest = load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as target:
        for record in manifest:
            source_path = Path(record.pop("_path"))
            try:
                text = source_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Cannot read source {source_path}: {exc}") from exc
            metadata = {key: record[key] for key in REQUIRED_FIELDS - {"path"}}
            metadata["published_date"] = record.get("published_date")
            chunks = chunk_text(text, metadata, chunk_words, overlap_words)
            for chunk in chunks:
                target.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
                count += 1
    return count


def read_chunks(path: Path, source_manifest_path: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    sources: dict[str, dict[str, str]] = {}
    if source_manifest_path:
        try:
            source_records = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read source metadata {source_manifest_path}: {exc}") from exc
        if not isinstance(source_records, list):
            raise ValueError("source metadata must be a JSON list")
        sources = {str(row["document_id"]): row for row in source_records}
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if REQUIRED_FIELDS - {"path"} <= record.keys():
                        chunks.append(Chunk(**record))
                    elif {"chunk_id", "source_doc", "text"} <= record.keys():
                        source_doc = str(record["source_doc"])
                        metadata = sources.get(source_doc)
                        if metadata is None:
                            raise ValueError(
                                f"Pre-chunked record references {source_doc!r}; "
                                "provide --source-manifest with provenance metadata"
                            )
                        chunks.append(Chunk(
                            chunk_id=str(record["chunk_id"]),
                            document_id=source_doc,
                            title=str(metadata["title"]),
                            publisher=str(metadata["publisher"]),
                            url=str(metadata["url"]),
                            accessed_date=str(metadata.get("accessed_date") or "unknown"),
                            license_note=str(metadata["license_note"]),
                            section=str(record.get("section_heading") or f"Page {record.get('page', 'unknown')}"),
                            text=str(record["text"]),
                            published_date=metadata.get("published_date"),
                        ))
                    else:
                        raise TypeError("unrecognized chunk schema")
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"Malformed chunk record at {path}:{line_number}") from exc
                except ValueError as exc:
                    raise ValueError(f"Invalid chunk record at {path}:{line_number}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read corpus {path}: {exc}") from exc
    return chunks
