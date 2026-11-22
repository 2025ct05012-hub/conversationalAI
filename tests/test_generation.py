import json
from io import BytesIO
import unittest
from unittest.mock import patch

from rag.generation import (
    generate_answer,
    generate_answer_stream,
    generate_baseline_answer,
)


class GenerationTests(unittest.TestCase):
    def test_baseline_calls_model_without_document_context(self):
        with patch(
            "rag.generation._chat_completion", return_value="Baseline response"
        ) as chat:
            answer = generate_baseline_answer("What is RAG?", "local-model")

        self.assertEqual(answer, "Baseline response")
        messages = chat.call_args.args[1]
        self.assertIn("pretrained knowledge", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "What is RAG?")

    def test_missing_evidence_does_not_call_the_model(self):
        with patch("rag.generation._chat_completion") as chat:
            answer = generate_answer("Question", [], "local-model")

        self.assertIn("could not find", answer)
        chat.assert_not_called()

    def test_answer_stream_yields_ollama_tokens_incrementally(self):
        events = b"".join(
            json.dumps(
                {"message": {"content": part}, "done": done}
            ).encode("utf-8")
            + b"\n"
            for part, done in [("Grounded ", False), ("answer.", True)]
        )
        context = [
            {
                "source": "lecture.pdf",
                "page": 2,
                "chunk_index": 0,
                "text": "Supporting evidence.",
            }
        ]
        with patch("rag.generation.urlopen", return_value=BytesIO(events)):
            parts = list(
                generate_answer_stream("Question?", context, "local-model")
            )

        self.assertEqual(parts, ["Grounded ", "answer."])


if __name__ == "__main__":
    unittest.main()
