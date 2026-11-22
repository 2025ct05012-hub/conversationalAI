"""Split page text into overlapping embedding-token chunks."""

from typing import Any, Dict, List, Optional


def tokenize_ids(tokenizer: Any, text: str) -> List[Any]:
    """Tokenize long text without triggering model-length warnings."""
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        return backend_tokenizer.encode(
            text, add_special_tokens=False
        ).ids

    tokenize = getattr(tokenizer, "tokenize", None)
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(tokenize) and callable(convert_tokens_to_ids):
        return convert_tokens_to_ids(tokenize(text))
    return tokenizer.encode(text, add_special_tokens=False)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 60,
    tokenizer: Optional[Any] = None,
) -> List[str]:
    """Split text by a model tokenizer, or whitespace tokens for unit tests.

    Pass the embedding model tokenizer in production so configured sizes refer
    to actual model tokens. The whitespace fallback keeps pure unit tests
    independent of downloaded models.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text.strip():
        return []

    if tokenizer is None:
        tokens = text.split()
        decode = lambda values: " ".join(values)
    else:
        tokens = tokenize_ids(tokenizer, text)
        decode = lambda values: tokenizer.decode(
            values, skip_special_tokens=True
        ).strip()

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = decode(tokens[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def chunk_pages(
    pages: List[Dict[str, object]],
    chunk_size: int = 500,
    overlap: int = 60,
    tokenizer: Optional[Any] = None,
) -> List[Dict[str, object]]:
    """Chunk extracted pages while retaining their source and page metadata."""
    chunks: List[Dict[str, object]] = []
    for page in pages:
        for chunk_index, text in enumerate(
            chunk_text(
                str(page["text"]), chunk_size, overlap, tokenizer=tokenizer
            )
        ):
            chunks.append(
                {
                    "text": text,
                    "source": str(page["source"]),
                    "page": int(page["page"]),
                    "chunk_index": chunk_index,
                }
            )
    return chunks
