import unittest

from rag.embeddings import encode_chunks


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class FakeEmbeddingModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 4

    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        self.encoded_texts.extend(texts)
        return [[float(index + 1)] for index, _ in enumerate(texts)]


class EmbeddingWindowTests(unittest.TestCase):
    def test_long_chunks_encode_every_window_then_average(self):
        model = FakeEmbeddingModel()
        vectors = encode_chunks(model, ["a b c d e"])
        self.assertEqual(model.encoded_texts, ["a b", "c d", "e"])
        self.assertEqual(vectors.shape, (1, 1))
        self.assertAlmostEqual(float(vectors[0][0]), 2.0)


if __name__ == "__main__":
    unittest.main()
