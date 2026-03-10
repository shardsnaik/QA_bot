"""
LLM client wrapping the Groq API for fast inference.
"""

from __future__ import annotations

import logging
from groq import Groq
from utils.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "You are a helpful and versatile assistant. "
    "If context is provided below, prioritize it to answer the question accurately. "
    "If no context is provided or the context is irrelevant, use your own general knowledge to answer. "
    "Always specify if you are answering based on the provided documents or your general knowledge."
)

ANSWER_TEMPLATE = (
    "Context (if any):\n"
    "─────────────────────────────────\n"
    "{context}\n"
    "─────────────────────────────────\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def generate_answer(context: str, question: str) -> str:
    """
    Send the context + question to the Groq LLM and return the generated answer.

    Parameters
    ----------
    context : str
        Retrieved text chunks concatenated together.
    question : str
        The user's question.

    Returns
    -------
    str
        The LLM-generated answer.
    """
    client = _get_client()

    user_message = ANSWER_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    logger.info("LLM generated answer (model=%s, tokens=%d)", LLM_MODEL, response.usage.total_tokens)
    return answer
