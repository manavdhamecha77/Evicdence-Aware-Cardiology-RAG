import argparse
import json
import os
from pathlib import Path

from .config import Config
from .evaluate import evaluate, load_questions
from .ingest import ingest, read_chunks
from .pipeline import QAPipeline
from .retrieval import create_index


def main() -> None:
    parser = argparse.ArgumentParser(prog="cardio-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="chunk local public educational sources")
    ingest_parser.add_argument("--manifest", type=Path, default=Path("data/sources/manifest.json"))
    ingest_parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    ask_parser = commands.add_parser("ask", help="answer one educational question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--variant", choices=("baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"), default="ea_rag")
    ask_parser.add_argument("--corpus", type=Path, default=Path("data/processed/chunks.jsonl"))
    ask_parser.add_argument("--source-manifest", type=Path)
    ask_parser.add_argument("--retrieval", choices=("lexical", "dense"), default=os.getenv("CARDIO_RAG_RETRIEVAL", "lexical"))
    ask_parser.add_argument("--embedding-model", default=os.getenv("CARDIO_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ask_parser.add_argument("--embedding-revision", default=os.getenv("CARDIO_RAG_EMBEDDING_REVISION"))
    eval_parser = commands.add_parser("evaluate", help="run baseline and EA-RAG on the same JSONL questions")
    eval_parser.add_argument("--questions", type=Path, required=True)
    eval_parser.add_argument("--corpus", type=Path, default=Path("data/processed/chunks.jsonl"))
    eval_parser.add_argument("--source-manifest", type=Path)
    eval_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    eval_parser.add_argument("--run-id", default="local")
    eval_parser.add_argument("--retrieval", choices=("lexical", "dense"), default=os.getenv("CARDIO_RAG_RETRIEVAL", "lexical"))
    eval_parser.add_argument("--embedding-model", default=os.getenv("CARDIO_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    eval_parser.add_argument("--embedding-revision", default=os.getenv("CARDIO_RAG_EMBEDDING_REVISION"))
    eval_parser.add_argument("--sample-size", type=int, help="deterministically sample this many questions")
    eval_parser.add_argument("--seed", type=int, default=17)
    eval_parser.add_argument("--include-options-in-query", action="store_true")
    eval_parser.add_argument("--min-score", type=float, default=0.9256)
    eval_parser.add_argument("--dense-min-score", type=float, default=0.45)
    eval_parser.add_argument("--retry-min-score", type=float, default=0.9294)
    eval_parser.add_argument("--dense-retry-min-score", type=float, default=0.47)
    eval_parser.add_argument("--variant", choices=("fixed", "baseline", "ea_rag", "ea_rag_no_abstention", "ea_rag_no_adaptation"))
    args = parser.parse_args()
    if args.command == "ingest":
        print(json.dumps({"chunks_written": ingest(args.manifest, args.output)}))
        return
    corpus = read_chunks(args.corpus, getattr(args, "source_manifest", None))
    config = Config(
        corpus_path=args.corpus,
        output_dir=getattr(args, "output", Path("artifacts")),
        source_manifest_path=getattr(args, "source_manifest", None),
        retrieval_backend=args.retrieval,
        embedding_model=args.embedding_model,
        embedding_revision=args.embedding_revision,
        include_options_in_query=getattr(args, "include_options_in_query", False),
        min_score=getattr(args, "min_score", 0.9256),
        retry_min_score=getattr(args, "retry_min_score", 0.9294),
        dense_min_score=getattr(args, "dense_min_score", 0.45),
        dense_retry_min_score=getattr(args, "dense_retry_min_score", 0.47),
        seed=getattr(args, "seed", 17),
    )
    pipeline = QAPipeline(create_index(corpus, config), config)
    if args.command == "ask":
        print(json.dumps(pipeline.answer(args.question, args.variant).to_dict(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(evaluate(
            pipeline, load_questions(args.questions), args.output, args.run_id,
            sample_size=args.sample_size, seed=args.seed,
            variants=(("baseline" if args.variant == "fixed" else args.variant),) if args.variant else None,
        ), indent=2))


if __name__ == "__main__":
    main()
