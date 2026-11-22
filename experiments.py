"""Run the assignment's chunking, retrieval, and RAG-vs-baseline experiments."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_pages
from rag.embeddings import encode_chunks
from rag.generation import generate_answer, generate_baseline_answer
from rag.pdf_processing import extract_pdf_pages
from rag.retrieval import BM25Retriever, FaissRetriever


CHUNK_SIZES = (200, 500, 1000)


def _load_pages(
    pdf_dir: Path, cache_path: Path
) -> List[Dict[str, object]]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}. Add authorized PDFs and retry."
        )
    fingerprint = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in pdf_paths
    ]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            print(f"Using cached extracted pages from {cache_path}.", flush=True)
            return cached["pages"]

    pages: List[Dict[str, object]] = []
    for pdf_path in pdf_paths:
        pages.extend(extract_pdf_pages(str(pdf_path)))
    temporary_cache = cache_path.with_suffix(".tmp")
    temporary_cache.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "pages": pages},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_cache.replace(cache_path)
    print(f"Cached extracted pages at {cache_path}.", flush=True)
    return pages


def _retrieval_metrics(
    question: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    expected_sources = set(question.get("expected_sources", []))
    expected_pages = {
        int(page) for page in question.get("expected_pages", [])
    }
    has_labels = bool(expected_sources or expected_pages)
    matches = [
        index
        for index, result in enumerate(results, start=1)
        if (
            (not expected_sources or result["source"] in expected_sources)
            and (not expected_pages or int(result["page"]) in expected_pages)
        )
    ]
    precision = len(matches) / top_k
    return {
        "expected_labels_provided": has_labels,
        "hit_rate_at_k": float(bool(matches)) if has_labels else "",
        "precision_at_k": precision if has_labels else "",
        "reciprocal_rank": (1 / matches[0]) if matches and has_labels else (
            0.0 if has_labels else ""
        ),
    }


def run_experiments(
    pdf_dir: Path,
    questions_path: Path,
    output_dir: Path,
    top_k: int = 5,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ollama_model: str = "llama3.2:3b",
    answer_limit: Optional[int] = None,
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")
    if answer_limit is not None and answer_limit < 1:
        raise ValueError("answer_limit must be greater than zero.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = _load_pages(pdf_dir, output_dir / "extracted_pages.json")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("The questions file must contain a non-empty JSON list.")

    model = SentenceTransformer(embedding_model_name)
    retrieval_rows: List[Dict[str, Any]] = []
    contexts_by_question: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for chunk_size in CHUNK_SIZES:
        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=min(60, chunk_size - 1),
            tokenizer=model.tokenizer,
        )
        if not chunks:
            raise ValueError(f"No chunks were created for size {chunk_size}.")
        vectors = encode_chunks(
            model, [str(chunk["text"]) for chunk in chunks]
        )
        dense = FaissRetriever(chunks, vectors)
        keyword = BM25Retriever(chunks)

        for question_number, question in enumerate(questions, start=1):
            query = str(question.get("question", "")).strip()
            if not query:
                raise ValueError(
                    f"Question {question_number} is missing its question text."
                )
            for method, retriever, args in (
                ("semantic_faiss", dense, (query, model, top_k)),
                ("keyword_bm25", keyword, (query, top_k)),
            ):
                started = time.perf_counter()
                results = retriever.search(*args)
                elapsed = time.perf_counter() - started
                retrieval_rows.append(
                    {
                        "chunk_size_tokens": chunk_size,
                        "method": method,
                        "question": query,
                        "top_k": top_k,
                        "latency_seconds": elapsed,
                        "result_count": len(results),
                        **_retrieval_metrics(question, results, top_k),
                    }
                )

            contexts_by_question[(chunk_size, question_number)] = dense.search(
                query, model, top_k
            )

    retrieval_path = output_dir / "retrieval_metrics.csv"
    summary_path = output_dir / "retrieval_summary.csv"
    answer_path = output_dir / "answer_comparison.csv"
    retrieval_frame = pd.DataFrame(retrieval_rows)
    retrieval_frame.to_csv(retrieval_path, index=False)
    score_columns = [
        "hit_rate_at_k",
        "precision_at_k",
        "reciprocal_rank",
        "latency_seconds",
    ]
    for column in score_columns:
        retrieval_frame[column] = pd.to_numeric(
            retrieval_frame[column], errors="coerce"
        )
    summary = (
        retrieval_frame.groupby(["chunk_size_tokens", "method"], as_index=False)
        .agg(
            questions=("question", "count"),
            labeled_questions=("expected_labels_provided", "sum"),
            hit_rate_at_k=("hit_rate_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    print(f"Retrieval metrics: {retrieval_path}")
    print(f"Retrieval summary: {summary_path}")
    print(f"Answer comparisons: {answer_path}")

    answer_rows: List[Dict[str, Any]] = []
    pd.DataFrame(
        columns=[
            "chunk_size_tokens",
            "question",
            "baseline_answer",
            "rag_answer",
            "rag_sources",
            "baseline_correctness_1_to_5",
            "rag_correctness_1_to_5",
            "baseline_relevance_1_to_5",
            "rag_relevance_1_to_5",
            "baseline_grounding_1_to_5",
            "rag_grounding_1_to_5",
            "reviewer_notes",
        ]
    ).to_csv(answer_path, index=False)
    answer_questions = (
        questions if answer_limit is None else questions[:answer_limit]
    )
    for question_number, question in enumerate(answer_questions, start=1):
        query = str(question["question"]).strip()
        print(
            f"Comparing baseline and RAG for question {question_number}/"
            f"{len(answer_questions)} at chunk size 500: {query}",
            flush=True,
        )
        baseline_answer = generate_baseline_answer(
            query, ollama_model, temperature=0.0, max_tokens=96
        )
        semantic_results = contexts_by_question[(500, question_number)]
        rag_answer = generate_answer(
            query,
            semantic_results,
            ollama_model,
            temperature=0.0,
            max_tokens=96,
        )
        answer_rows.append(
            {
                "chunk_size_tokens": 500,
                "question": query,
                "baseline_answer": baseline_answer,
                "rag_answer": rag_answer,
                "rag_sources": "; ".join(
                    "{} p. {}".format(item["source"], item["page"])
                    for item in semantic_results
                ),
                "baseline_correctness_1_to_5": "",
                "rag_correctness_1_to_5": "",
                "baseline_relevance_1_to_5": "",
                "rag_relevance_1_to_5": "",
                "baseline_grounding_1_to_5": "",
                "rag_grounding_1_to_5": "",
                "reviewer_notes": "",
            }
        )
        pd.DataFrame(answer_rows).to_csv(answer_path, index=False)

    print(
        "Human quality scores are blank by design; review the answer pairs "
        "against the source PDFs before reporting results."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation_questions.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument(
        "--answer-limit",
        type=int,
        default=None,
        help="Limit manual quality comparisons to the first N questions.",
    )
    args = parser.parse_args()
    if not args.questions.is_file():
        parser.error(
            f"Question file not found: {args.questions}. Copy "
            "data/evaluation_questions.example.json to "
            "data/evaluation_questions.json and edit it first."
        )
    run_experiments(
        args.pdf_dir,
        args.questions,
        args.output_dir,
        args.top_k,
        args.embedding_model,
        args.ollama_model,
        args.answer_limit,
    )


if __name__ == "__main__":
    main()
