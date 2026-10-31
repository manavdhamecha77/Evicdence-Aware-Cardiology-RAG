from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    corpus_path: Path = Path("data/processed/chunks.jsonl")
    output_dir: Path = Path("artifacts")
    source_manifest_path: Path | None = None
    top_k: int = 4
    # Frozen from the top-1 score distribution on a seeded 100-item D2 val sample.
    min_score: float = 0.9256
    retry_min_score: float = 0.9294
    min_passages: int = 1
    chunk_words: int = 550
    overlap_words: int = 80
    seed: int = 17
    retrieval_backend: str = "lexical"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str | None = None
    embedding_cache_dir: Path = Path("artifacts/indices")
    dense_min_score: float = 0.45
    dense_retry_min_score: float = 0.47
    include_options_in_query: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.min_passages < 1:
            raise ValueError("top_k and min_passages must be positive")
        if self.chunk_words < 1 or not 0 <= self.overlap_words < self.chunk_words:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        if self.retrieval_backend not in {"lexical", "dense"}:
            raise ValueError("retrieval_backend must be 'lexical' or 'dense'")
        if not 0 <= self.dense_min_score <= 1:
            raise ValueError("dense_min_score must be between zero and one")
        if not 0 <= self.retry_min_score <= 1 or not 0 <= self.dense_retry_min_score <= 1:
            raise ValueError("retry thresholds must be between zero and one")

    @property
    def evidence_min_score(self) -> float:
        return self.dense_min_score if self.retrieval_backend == "dense" else self.min_score

    @property
    def retry_trigger_min_score(self) -> float:
        return self.dense_retry_min_score if self.retrieval_backend == "dense" else self.retry_min_score
