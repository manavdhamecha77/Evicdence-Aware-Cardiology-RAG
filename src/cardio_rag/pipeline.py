import time
import re

from .citations import validate_citations, validate_mcq_response
from .answer_parsing import parse_mcq_answer
from .config import Config
from .evidence_gate import assess_evidence
from .generation import GenerationError, generator_from_environment
from .query_router import adapt_query, classify_intent
from .retrieval import DenseIndex, LexicalIndex
from .schemas import RunRecord

RESEARCH_NOTICE = "Research prototype for educational use only; not medical advice or a clinical decision system."
PERSONAL_SYMPTOM_NOTICE = "If these are your symptoms, contact a qualified healthcare professional; for urgent or severe symptoms, contact local emergency services. This prototype cannot assess personal symptoms."


class QAPipeline:
    def __init__(self, index: LexicalIndex | DenseIndex, config: Config | None = None):
        self.index = index
        self.config = config or Config()
        try:
            self.generator = generator_from_environment()
            self.generator_error: str | None = None
        except GenerationError as exc:
            self.generator = None
            self.generator_error = str(exc)

    def answer(
        self, question: str, variant: str = "baseline", question_id: str = "adhoc",
        reference_answer: str | None = None, relevant_chunk_ids: list[str] | None = None,
        options: dict[str, str] | None = None, evaluation_mode: bool = False,
    ) -> RunRecord:
        started = time.perf_counter()
        question = question.strip()
        intent = classify_intent(question)
        if not question:
            return RunRecord(question_id, question, variant, "Enter an educational question.", True, intent, abstention_reason="Empty question", latency_ms=(time.perf_counter() - started) * 1000)
        if not evaluation_mode and re.search(r"\b(i|i'm|i am|my|we|my child)\b.*\b(chest pain|shortness of breath|palpitation|faint|dizz|heart attack|symptom)\b", question.lower()):
            return RunRecord(question_id, question, variant, PERSONAL_SYMPTOM_NOTICE, True, intent, abstention_reason="Personal symptom safety response", latency_ms=(time.perf_counter() - started) * 1000)

        retrieval_query = question
        if options and self.config.include_options_in_query:
            retrieval_query += "\nOptions: " + " ".join(f"{key} {value}" for key, value in options.items())
        retrieved = self.index.retrieve(retrieval_query, self.config.top_k)
        initial_retrieved = list(retrieved)
        retry_count = 0
        initial_assessment = assess_evidence(retrieved, self.config.evidence_min_score, self.config.min_passages)
        retry_trigger = assess_evidence(retrieved, self.config.retry_trigger_min_score, self.config.min_passages)
        assessment = initial_assessment
        if variant in {"ea_rag", "ea_rag_no_abstention"} and not retry_trigger.sufficient:
            retry_count = 1
            retry_query = adapt_query(retrieval_query, intent)
            retry = self.index.retrieve(retry_query, self.config.top_k)
            merged: dict[str, object] = {item.chunk.chunk_id: item for item in retrieved}
            for item in retry:
                if item.chunk.chunk_id not in merged or item.score > merged[item.chunk.chunk_id].score:
                    merged[item.chunk.chunk_id] = item
            retrieved = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.chunk_id))[:self.config.top_k]
            assessment = assess_evidence(retrieved, self.config.evidence_min_score, self.config.min_passages)

        abstain = variant in {"ea_rag", "ea_rag_no_adaptation"} and not assessment.sufficient
        abstention_reason = assessment.reason if abstain else None
        predicted_answer: str | None = None
        correct: bool | None = None
        if abstain:
            answer = "The available indexed sources are insufficient to answer this question reliably. " + RESEARCH_NOTICE
            citations: list[str] = []
            citation_valid = False
        else:
            if variant in {"baseline", "ea_rag_no_abstention"}:
                generation_results = retrieved
            else:
                generation_results = [item for item in retrieved if item.score >= self.config.evidence_min_score]
            if self.generator_error:
                return RunRecord(
                    question_id=question_id, question=question, variant=variant,
                    answer=f"Answer generation failed: {self.generator_error}", abstained=True, intent=intent,
                    retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved],
                    retrieval_scores=[item.score for item in retrieved], retry_count=retry_count,
                    abstention_reason=self.generator_error,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    reference_answer=reference_answer, relevant_chunk_ids=relevant_chunk_ids or [],
                    metadata={"generator_error": self.generator_error, "run_error": True},
                )
            try:
                if options:
                    generated = self.generator.generate(question, generation_results, options)
                else:
                    generated = self.generator.generate(question, generation_results)
            except GenerationError as exc:
                return RunRecord(
                    question_id=question_id, question=question, variant=variant,
                    answer=f"Answer generation failed: {exc}", abstained=True, intent=intent,
                    retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved],
                    retrieval_scores=[item.score for item in retrieved], retry_count=retry_count,
                    abstention_reason=str(exc), latency_ms=(time.perf_counter() - started) * 1000,
                    reference_answer=reference_answer, relevant_chunk_ids=relevant_chunk_ids or [],
                    metadata={"generator_error": str(exc), "run_error": True, "provider": self.generator.provider, "model": self.generator.model},
                )
            answer = generated.answer
            predicted_answer = parse_mcq_answer(answer) if options else None
            correct = (predicted_answer == reference_answer.strip().upper()) if options and reference_answer else None
            if options:
                mcq_validation = validate_mcq_response(answer, generation_results)
                predicted_answer = mcq_validation["answer_letter"]
                citations = mcq_validation["cited_ids"]
                citation_valid = mcq_validation["citation_valid"]
            else:
                citation_valid, citations = validate_citations(answer, generation_results)
            if not options and not citations and variant in {"ea_rag", "ea_rag_no_adaptation"}:
                abstain = True
                abstention_reason = "No citable evidence passages were available."
                answer = "The available indexed sources are insufficient to answer this question reliably. " + RESEARCH_NOTICE
        if options and reference_answer:
            correct = predicted_answer == reference_answer.strip().upper() and not abstain

        metadata = {
            "evidence": assessment.signals,
            "initial_evidence": initial_assessment.signals,
            "initial_sufficient": initial_assessment.sufficient,
            "retry_threshold": self.config.retry_trigger_min_score,
            "generator": self.generator.provider if self.generator else "configuration_error",
            "model": self.generator.model if self.generator else None,
            "task": "mcq" if options else "open_qa",
            "primary_latency_ms": (time.perf_counter() - started) * 1000,
        }
        # Evaluation traces preserve the counterfactual answers needed to derive
        # ablations without repeating every model call. These candidates are not
        # exposed as the EA-RAG answer when the final evidence gate abstains.
        if evaluation_mode and variant == "ea_rag" and options and not self.generator_error:
            def as_candidate(items):
                chosen = [item for item in items if item.score >= self.config.evidence_min_score]
                if not chosen:
                    chosen = items
                candidate_started = time.perf_counter()
                try:
                    result = self.generator.generate(question, chosen, options)
                except GenerationError:
                    return None
                letter = parse_mcq_answer(result.answer)
                return {
                    "answer": result.answer,
                    "predicted_answer": letter,
                    "correct": letter == reference_answer.strip().upper() if reference_answer and letter else False,
                    "latency_ms": (time.perf_counter() - candidate_started) * 1000,
                }

            if abstain:
                metadata["counterfactual_no_abstention"] = as_candidate(retrieved)
            if retry_count and initial_assessment.sufficient:
                metadata["counterfactual_no_adaptation"] = as_candidate(initial_retrieved)

        return RunRecord(
            question_id=question_id, question=question, variant=variant, answer=answer, abstained=abstain,
            intent=intent, retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved],
            retrieval_scores=[item.score for item in retrieved], citations=citations, citation_valid=citation_valid,
            retry_count=retry_count, abstention_reason=abstention_reason,
            latency_ms=(time.perf_counter() - started) * 1000, reference_answer=reference_answer,
            predicted_answer=predicted_answer, correct=correct,
            relevant_chunk_ids=relevant_chunk_ids or [], metadata=metadata,
        )
