from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    publisher: str
    url: str
    accessed_date: str
    license_note: str
    section: str
    text: str
    published_date: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RunRecord:
    question_id: str
    question: str
    variant: str
    answer: str
    abstained: bool
    intent: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    citation_valid: bool = False
    retry_count: int = 0
    abstention_reason: str | None = None
    latency_ms: float = 0.0
    reference_answer: str | None = None
    relevant_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    predicted_answer: str | None = None
    correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
