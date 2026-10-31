import re


_FINAL_MCQ_ANSWER = re.compile(r"^\s*FINAL ANSWER\s*:\s*([A-D])\s*[.!]?\s*$", re.IGNORECASE | re.MULTILINE)


def parse_mcq_answer(response: str) -> str | None:
    """Return a choice only when the response has one unambiguous final-answer line."""
    matches = _FINAL_MCQ_ANSWER.findall(response)
    return matches[0].upper() if len(matches) == 1 else None
