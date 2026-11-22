import unittest

from rag.chunking import chunk_pages, chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunks_have_requested_size_and_overlap(self):
        chunks = chunk_text("one two three four five six", chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five", "five six"])

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("  "), [])

    def test_invalid_chunk_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=2, overlap=2)

    def test_page_metadata_is_preserved(self):
        chunks = chunk_pages(
            [{"source": "lecture.pdf", "page": 4, "text": "alpha beta"}],
            chunk_size=1,
            overlap=0,
        )
        self.assertEqual(
            [(chunk["source"], chunk["page"]) for chunk in chunks],
            [("lecture.pdf", 4), ("lecture.pdf", 4)],
        )


if __name__ == "__main__":
    unittest.main()
