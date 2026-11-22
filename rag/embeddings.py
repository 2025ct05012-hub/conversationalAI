"""Embedding helpers that preserve text beyond a model's context window."""

from typing import Any, List, Sequence

import numpy as np

from rag.chunking import tokenize_ids


def encode_chunks(embedding_model: Any, texts: Sequence[str]) -> np.ndarray:
    """Encode all tokens in every chunk by averaging window embeddings.

    Sentence-transformer models have a fixed maximum sequence length. Without
    windowing, long experimental chunks would be silently truncated and their
    trailing evidence would never reach the FAISS index.
    """
    tokenizer = embedding_model.tokenizer
    max_length = int(embedding_model.max_seq_length)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    window_size = max_length - special_tokens - 32
    if window_size < 1:
        window_size = max_length - special_tokens
    if window_size < 1:
        raise ValueError("Embedding model context window is too small.")

    windows: List[str] = []
    owners: List[int] = []
    for owner, text in enumerate(texts):
        token_ids = tokenize_ids(tokenizer, text)
        if not token_ids:
            raise ValueError("Cannot embed an empty text chunk.")
        start = 0
        while start < len(token_ids):
            end = min(start + window_size, len(token_ids))
            while end > start:
                window_text = tokenizer.decode(
                    token_ids[start:end], skip_special_tokens=True
                )
                encoded_length = len(tokenize_ids(tokenizer, window_text))
                overflow = encoded_length + special_tokens - max_length
                if overflow <= 0:
                    break
                end = max(start + 1, end - overflow)
            if end <= start:
                raise ValueError(
                    "The embedding tokenizer cannot fit a text window in "
                    "the model context length."
                )
            windows.append(window_text)
            owners.append(owner)
            start = end

    window_vectors = np.asarray(
        embedding_model.encode(
            windows, convert_to_numpy=True, show_progress_bar=False
        ),
        dtype="float32",
    )
    if window_vectors.ndim != 2 or window_vectors.shape[0] != len(windows):
        raise ValueError("The embedding model returned an invalid vector shape.")

    chunk_vectors = np.zeros(
        (len(texts), window_vectors.shape[1]), dtype="float32"
    )
    counts = np.zeros(len(texts), dtype="float32")
    for owner, vector in zip(owners, window_vectors):
        chunk_vectors[owner] += vector
        counts[owner] += 1
    chunk_vectors /= counts[:, None]
    return chunk_vectors
