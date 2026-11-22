"""Grounded chat completions using the local Ollama HTTP API."""

import json
import os
from typing import Any, Dict, Iterator, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def list_ollama_models() -> List[str]:
    request = Request(f"{OLLAMA_URL}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Start Ollama and retry. "
            f"Details: {error}"
        ) from error
    return sorted(
        model["name"] for model in payload.get("models", []) if "name" in model
    )


def generate_answer(
    question: str,
    context: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 128,
) -> str:
    """Answer only from supplied chunks; an empty context is a clear refusal."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not context:
        return "I could not find supporting information in the indexed PDFs."

    evidence = "\n\n".join(
        "[{} p. {} | chunk {}]\n{}".format(
            item["source"],
            item["page"],
            item.get("chunk_index", 0) + 1,
            item["text"],
        )
        for item in context
    )
    return _chat_completion(
        model,
        _grounded_messages(question, evidence),
        temperature,
        max_tokens,
    )


def generate_answer_stream(
    question: str,
    context: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 128,
) -> Iterator[str]:
    """Yield answer text as Ollama generates it."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not context:
        yield "I could not find supporting information in the indexed PDFs."
        return

    evidence = "\n\n".join(
        "[{} p. {} | chunk {}]\n{}".format(
            item["source"],
            item["page"],
            item.get("chunk_index", 0) + 1,
            item["text"],
        )
        for item in context
    )
    yield from _stream_chat_completion(
        model,
        _grounded_messages(question, evidence),
        temperature,
        max_tokens,
    )


def _grounded_messages(question: str, evidence: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using only the supplied evidence. "
                "If the evidence does not support an answer, say so. "
                "Do not invent facts or citations. Cite supporting chunks "
                "in the form [filename p. N]. Answer concisely in at most "
                "two or three sentences."
            ),
        },
        {
            "role": "user",
            "content": f"Evidence:\n{evidence}\n\nQuestion: {question}",
        },
    ]


def generate_baseline_answer(
    question: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 128,
) -> str:
    """Ask the model without document context for the no-RAG baseline."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    return _chat_completion(
        model,
        [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using your pretrained knowledge. "
                    "No course documents or retrieved evidence are available. "
                    "Do not claim that you consulted a source document. Answer "
                    "concisely in at most two or three sentences."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature,
        max_tokens,
    )


def _chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than zero.")
    payload = {
        "model": model,
        "keep_alive": "30m",
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": messages,
    }
    request = Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Ollama request failed for model '{model}'. Check that the model "
            f"is installed and Ollama is running. Details: {error}"
        ) from error

    answer = result.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")
    return answer


def _stream_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Iterator[str]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than zero.")
    payload = {
        "model": model,
        "keep_alive": "30m",
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": messages,
    }
    request = Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            received_text = False
            for line in response:
                if not line.strip():
                    continue
                result = json.loads(line.decode("utf-8"))
                if result.get("error"):
                    raise RuntimeError(
                        f"Ollama error for model '{model}': {result['error']}"
                    )
                content = result.get("message", {}).get("content", "")
                if content:
                    received_text = True
                    yield content
                if result.get("done"):
                    break
            if not received_text:
                raise RuntimeError("Ollama returned an empty answer.")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Ollama request failed for model '{model}'. Check that the model "
            f"is installed and Ollama is running. Details: {error}"
        ) from error
