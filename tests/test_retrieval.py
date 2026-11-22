import unittest

import numpy as np

from rag.retrieval import (
    MIN_RELEVANCE_SCORE,
    BM25Retriever,
    FaissRetriever,
    has_relevant_evidence,
)


class BM25Tests(unittest.TestCase):
    def test_keyword_match_ranks_relevant_chunk_first(self):
        chunks = [
            {"text": "photosynthesis converts light into chemical energy", "page": 1},
            {"text": "the water cycle includes evaporation and rain", "page": 2},
        ]
        results = BM25Retriever(chunks).search("photosynthesis light energy")
        self.assertEqual(results[0]["page"], 1)
        self.assertGreater(results[0]["score"], 0)

    def test_no_matching_terms_returns_empty(self):
        chunks = [{"text": "linear algebra matrices", "page": 1}]
        self.assertEqual(BM25Retriever(chunks).search("ocean currents"), [])

    def test_empty_collection_and_invalid_k_are_rejected(self):
        with self.assertRaises(ValueError):
            BM25Retriever([])
        retriever = BM25Retriever([{"text": "a document"}])
        with self.assertRaises(ValueError):
            retriever.search("document", top_k=0)


class FaissTests(unittest.TestCase):
    def test_normalized_vector_search_returns_nearest_chunk(self):
        chunks = [
            {"text": "first", "page": 1},
            {"text": "second", "page": 2},
        ]
        retriever = FaissRetriever(
            chunks, np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        )

        class QueryEncoder:
            def encode(self, texts):
                return np.array([[0.9, 0.1]], dtype="float32")

        results = retriever.search("query", QueryEncoder(), top_k=1)
        self.assertEqual(results[0]["page"], 1)

    def test_embedding_rows_must_match_chunks(self):
        with self.assertRaises(ValueError):
            FaissRetriever([{"text": "one"}], np.zeros((0, 2), dtype="float32"))

    def test_weak_nearest_match_is_not_treated_as_evidence(self):
        self.assertFalse(
            has_relevant_evidence([{"score": MIN_RELEVANCE_SCORE - 0.01}])
        )
        self.assertFalse(has_relevant_evidence([]))

    def test_relevant_match_passes_relevance_threshold(self):
        self.assertTrue(
            has_relevant_evidence([{"score": MIN_RELEVANCE_SCORE}])
        )


if __name__ == "__main__":
    unittest.main()
