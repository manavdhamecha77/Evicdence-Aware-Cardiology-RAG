import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schemas import RetrievedChunk

FALLBACK_LABEL = "Extractive demo response (no language model configured)."


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    provider: str
    model: str


class Generator(Protocol):
    provider: str
    model: str

    def generate(
        self, question: str, results: list[RetrievedChunk], options: dict[str, str] | None = None
    ) -> GenerationResult: ...


def extractive_answer(results: list[RetrievedChunk], max_sentences: int = 3) -> tuple[str, list[str]]:
    if not results:
        return "The available sources do not provide enough evidence to answer this question.", []
    sentences: list[tuple[float, str, str]] = []
    for result in results:
        for sentence in result.chunk.text.replace("\n", " ").split(". "):
            cleaned = sentence.strip().rstrip(".")
            if len(cleaned.split()) >= 5:
                sentences.append((result.score, cleaned, result.chunk.chunk_id))
    selected = sorted(sentences, key=lambda item: (-item[0], item[2]))[:max_sentences]
    if not selected:
        selected = [(results[0].score, results[0].chunk.text.strip(), results[0].chunk.chunk_id)]
    answer = " ".join(f"{sentence}. [{chunk_id}]" for _, sentence, chunk_id in selected)
    return f"{FALLBACK_LABEL}\n\n{answer}", [chunk_id for _, _, chunk_id in selected]


class ExtractiveGenerator:
    provider = "extractive"
    model = "deterministic-extractive-v1"

    def generate(self, question: str, results: list[RetrievedChunk], options: dict[str, str] | None = None) -> GenerationResult:
        if options:
            raise GenerationError("The extractive fallback cannot answer multiple-choice questions.")
        answer, _ = extractive_answer(results)
        return GenerationResult(answer, self.provider, self.model)


class OpenAICompatibleGenerator:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 45.0):
        local_endpoint = base_url.startswith(("http://localhost", "http://127.0.0.1"))
        if (not api_key.strip() and not local_endpoint) or not model.strip():
            raise ValueError("A model name and either an API key or local Ollama endpoint are required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.provider = "ollama_openai_compatible" if "ollama.com" in self.base_url or "localhost:11434" in self.base_url else "openai_chat_completions"

    def generate(
        self, question: str, results: list[RetrievedChunk], options: dict[str, str] | None = None
    ) -> GenerationResult:
        context = "\n\n".join(
            f"[{item.chunk.chunk_id}] {item.chunk.title} | {item.chunk.publisher} | {item.chunk.url}\n{item.chunk.text}"
            for item in results
        )
        if options:
            option_text = "\n".join(f"{key}) {value}" for key, value in options.items())
            system_prompt = (
                "You are answering an educational cardiology multiple-choice benchmark. Use the supplied passages "
                "as evidence and choose exactly one of A, B, C, or D. Evidence-gated variants handle abstention "
                "before generation. Output exactly one line and nothing else, in this format: FINAL ANSWER: A. "
                "Do not include reasoning or citations."
            )
            user_prompt = (
                f"Question stem:\n{question}\n\nOptions:\n{option_text}\n\n"
                f"Source passages:\n{context}"
            )
        else:
            system_prompt = (
                "Answer educational cardiology questions only from the supplied source passages. "
                "Cite every factual claim with one or more exact retrieved chunk IDs in square brackets. "
                "If the passages do not support an answer, say the evidence is insufficient. "
                "Do not give personalized diagnosis or treatment advice. Keep the answer concise."
            )
            user_prompt = f"Question: {question}\n\nSource passages:\n{context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise GenerationError(f"Model API returned HTTP {exc.code}.") from exc
        except URLError as exc:
            raise GenerationError(f"Model API connection failed: {exc.reason}.") from exc
        except (TimeoutError, OSError) as exc:
            raise GenerationError(f"Model API request failed: {type(exc).__name__}.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GenerationError("Model API returned an invalid JSON response.") from exc
        try:
            answer = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError("Model API response did not contain a completion message.") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError("Model API returned an empty answer.")
        return GenerationResult(answer.strip(), self.provider, self.model)


def _dotenv_values(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    raw_values: dict[str, str] = {}
    assignment = re.compile(r"^(?:\$env:)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        match = assignment.fullmatch(line)
        if not match:
            raise GenerationError(f"Invalid .env entry on line {line_number}; use NAME=value assignments.")
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif value.startswith(("\"", "'")):
            value = value[1:]
        raw_values[name] = value
    values: dict[str, str] = {}
    for name, value in raw_values.items():
        if value.startswith("$env:"):
            reference = value[len("$env:"):].strip()
            value = os.environ.get(reference, raw_values.get(reference, ""))
            if not value:
                raise GenerationError(f".env reference for {name} points to an unset variable.")
        values[name] = value
    return values


def generator_from_environment() -> Generator:
    file_values = _dotenv_values()
    settings = {**file_values, **os.environ}
    ollama_key = settings.get("OLLAMA_API_KEY", "").strip()
    api_key = ollama_key or settings.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        local_model = settings.get("OLLAMA_MODEL", "").strip()
        local_base_url = (
            settings.get("OLLAMA_BASE_URL", "").strip()
            or settings.get("OLLAMA_HOST", "").strip()
        )
        if local_model and not local_base_url:
            local_base_url = "http://localhost:11434/v1"
        if local_model and local_base_url.startswith(("http://localhost", "http://127.0.0.1")):
            if not local_base_url.rstrip("/").endswith("/v1"):
                local_base_url = local_base_url.rstrip("/") + "/v1"
            return OpenAICompatibleGenerator("", local_model, local_base_url)
        if local_model and local_base_url.startswith("https://"):
            raise GenerationError("OLLAMA_API_KEY is required for the configured remote Ollama endpoint.")
        return ExtractiveGenerator()
    model = settings.get("OLLAMA_MODEL", "").strip() or settings.get("OPENAI_MODEL", "").strip()
    if not model:
        model_variable = "OLLAMA_MODEL" if ollama_key else "OPENAI_MODEL"
        raise GenerationError(f"{model_variable} must be set when a model API key is configured.")
    default_base = "https://ollama.com/v1" if ollama_key else "https://api.openai.com/v1"
    base_url = settings.get("OLLAMA_BASE_URL", "").strip() or settings.get("OPENAI_BASE_URL", default_base).strip()
    try:
        timeout = float(settings.get("OLLAMA_TIMEOUT_SECONDS", "").strip() or settings.get("OPENAI_TIMEOUT_SECONDS", "45"))
    except ValueError as exc:
        raise GenerationError("OPENAI_TIMEOUT_SECONDS must be a number.") from exc
    if timeout <= 0:
        raise GenerationError("OPENAI_TIMEOUT_SECONDS must be greater than zero.")
    return OpenAICompatibleGenerator(api_key, model, base_url, timeout)
