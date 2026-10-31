import re

from .answer_parsing import parse_mcq_answer
from .schemas import RetrievedChunk

CITATION = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def validate_citations(answer: str, retrieved: list[RetrievedChunk]) -> tuple[bool, list[str]]:
    citation_ids = CITATION.findall(answer)
    allowed = {item.chunk.chunk_id for item in retrieved}
    paragraphs = [part.strip() for part in answer.splitlines() if part.strip() and not part.startswith("Extractive demo response")]
    valid = bool(paragraphs) and all(CITATION.search(part) for part in paragraphs) and bool(citation_ids) and set(citation_ids) <= allowed
    return valid, citation_ids


def validate_mcq_response(answer: str, retrieved: list[RetrievedChunk]) -> dict[str, object]:
    """Validate optional MCQ citations without requiring citations for a bare choice."""
    cited_ids = CITATION.findall(answer)
    allowed = {item.chunk.chunk_id for item in retrieved}
    invalid_ids = [chunk_id for chunk_id in cited_ids if chunk_id not in allowed]
    answer_letter = parse_mcq_answer(answer)
    return {
        "answer_letter": answer_letter,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "citation_valid": not invalid_ids,
        "has_support": bool(answer_letter and retrieved),
    }
