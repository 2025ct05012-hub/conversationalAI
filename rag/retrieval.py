"""Dense FAISS and lexical BM25 retrievers."""

import math
import re
from typing import Any, Dict, List, Sequence, Tuple

MIN_RELEVANCE_SCORE = 0.30


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Small dependency-free BM25 ranker for an in-memory chunk collection."""

    def __init__(self, chunks: Sequence[Dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("Cannot create a retriever without document chunks.")
        self.chunks = list(chunks)
        self.terms = [_tokenize(str(chunk["text"])) for chunk in self.chunks]
        self.lengths = [len(document) for document in self.terms]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency: Dict[str, int] = {}
        for document in self.terms:
            for term in set(document):
                self.document_frequency[term] = (
                    self.document_frequency.get(term, 0) + 1
                )

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        k1 = 1.5
        b = 0.75
        scored: List[Tuple[float, int]] = []
        document_count = len(self.chunks)
        for index, terms in enumerate(self.terms):
            frequencies: Dict[str, int] = {}
            for term in terms:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1 + (document_count - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * self.lengths[index] / max(self.average_length, 1)
                )
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**self.chunks[index], "score": float(score)}
            for score, index in scored[:top_k]
            if score > 0
        ]


class FaissRetriever:
    """Cosine-similarity search backed by normalized FAISS inner product."""

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        embeddings: Any,
    ) -> None:
        import faiss
        import numpy as np

        if not chunks:
            raise ValueError("Cannot create a retriever without document chunks.")
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedding rows must match the number of chunks.")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._faiss = faiss
        self.chunks = list(chunks)

    def search(
        self,
        query: str,
        embedding_model: Any,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        query_vector = np.asarray(
            embedding_model.encode([query]), dtype="float32"
        )
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("The embedding model must return one query vector.")
        self._faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(
            query_vector, min(top_k, len(self.chunks))
        )
        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                results.append(
                    {**self.chunks[int(index)], "score": float(score)}
                )
        return results


def has_relevant_evidence(
    results: Sequence[Dict[str, Any]],
    minimum_score: float = MIN_RELEVANCE_SCORE,
) -> bool:
    """Reject weak semantic matches instead of treating every nearest chunk as evidence."""
    return bool(results and float(results[0]["score"]) >= minimum_score)
