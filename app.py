"""Streamlit interface for local academic-PDF question answering."""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_pages
from rag.embeddings import encode_chunks
from rag.generation import generate_answer_stream, list_ollama_models
from rag.pdf_processing import extract_pdf_pages
from rag.retrieval import (
    MIN_RELEVANCE_SCORE,
    BM25Retriever,
    FaissRetriever,
    has_relevant_evidence,
)


PROJECT_DIR = Path(__file__).resolve().parent
PDF_DIR = PROJECT_DIR / "data" / "pdfs"
PAGE_CACHE = PROJECT_DIR / "results" / "extracted_pages.json"


def _build_knowledge_base(
    pages: List[Dict[str, object]],
    embedding_model: SentenceTransformer,
    chunk_size: int,
) -> Dict[str, Any]:
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=min(60, chunk_size - 1),
        tokenizer=embedding_model.tokenizer,
    )
    if not chunks:
        raise ValueError("The PDFs did not produce any text chunks.")
    embeddings = encode_chunks(
        embedding_model, [str(chunk["text"]) for chunk in chunks]
    )
    return {
        "chunks": chunks,
        "dense": FaissRetriever(chunks, embeddings),
        "keyword": BM25Retriever(chunks),
        "chunk_size": chunk_size,
    }


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Preparing PDFs from data/pdfs...")
def load_folder_knowledge_base(
    pdf_paths: Tuple[str, ...],
    fingerprint: Tuple[Tuple[str, int, int], ...],
    embedding_name: str,
    chunk_size: int,
) -> Dict[str, Any]:
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. Add your PDFs to that folder."
        )

    cache_fingerprint = [
        {"name": name, "size": size, "modified_ns": modified_ns}
        for name, size, modified_ns in fingerprint
    ]
    pages: List[Dict[str, object]] = []
    if PAGE_CACHE.is_file():
        cached = json.loads(PAGE_CACHE.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == cache_fingerprint:
            pages = cached["pages"]

    if not pages:
        for pdf_path in pdf_paths:
            pages.extend(extract_pdf_pages(pdf_path))
        PAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = PAGE_CACHE.with_suffix(".tmp")
        temporary_cache.write_text(
            json.dumps(
                {"fingerprint": cache_fingerprint, "pages": pages},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_cache.replace(PAGE_CACHE)

    model = load_embedding_model(embedding_name)
    return _build_knowledge_base(pages, model, chunk_size)


def _index_uploads(
    uploaded_files: List[Any],
    embedding_model: SentenceTransformer,
    chunk_size: int,
) -> Dict[str, Any]:
    all_pages: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="academic_rag_") as temporary_dir:
        for uploaded in uploaded_files:
            path = Path(temporary_dir) / Path(uploaded.name).name
            path.write_bytes(uploaded.getvalue())
            all_pages.extend(extract_pdf_pages(str(path)))

    return _build_knowledge_base(all_pages, embedding_model, chunk_size)


st.set_page_config(page_title="Academic PDF Q&A", page_icon="📚", layout="wide")
st.title("Academic PDF Question Answering")
st.caption("Ask questions about your PDFs and inspect the retrieved evidence.")

embedding_name = os.environ.get(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
with st.sidebar:
    st.header("Knowledge base")
    selected_chunk_size = st.select_slider(
        "Chunk size (embedding tokens)",
        options=[200, 500, 1000],
        value=500,
    )
    st.caption(f"Embedding model: `{embedding_name}`")
    retrieval_method = st.radio(
        "Retrieval method", ["Semantic (FAISS)", "Keyword (BM25)"]
    )
    top_k = st.slider("Evidence chunks", min_value=1, max_value=10, value=5)

    with st.expander("Optional: use different PDFs"):
        uploads = st.file_uploader(
            "Upload academic PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Process uploaded PDFs", disabled=not uploads):
            try:
                model = load_embedding_model(embedding_name)
                with st.spinner("Extracting, chunking, and indexing PDFs..."):
                    st.session_state["knowledge_base"] = _index_uploads(
                        uploads, model, selected_chunk_size
                    )
                    st.session_state["knowledge_source"] = "uploads"
                    st.session_state["embedding_name"] = embedding_name
                st.success(
                    f"Indexed {len(st.session_state['knowledge_base']['chunks'])} "
                    f"chunks from {len(uploads)} PDF(s)."
                )
            except (RuntimeError, ValueError, OSError) as error:
                st.error(f"Could not process uploaded PDFs: {error}")

        if st.session_state.get("knowledge_source") == "uploads":
            if st.button("Return to PDFs in data/pdfs"):
                st.session_state.pop("knowledge_base", None)
                st.session_state["knowledge_source"] = "folder"
                st.session_state.pop("folder_index_key", None)
                st.rerun()

folder_pdfs = tuple(
    sorted(str(path.resolve()) for path in PDF_DIR.glob("*.pdf"))
)
folder_fingerprint = tuple(
    (
        Path(path).name,
        Path(path).stat().st_size,
        Path(path).stat().st_mtime_ns,
    )
    for path in folder_pdfs
)
folder_index_key = (
    folder_pdfs,
    folder_fingerprint,
    embedding_name,
    selected_chunk_size,
)

if st.session_state.get("knowledge_source") != "uploads":
    if st.session_state.get("folder_index_key") != folder_index_key:
        try:
            st.session_state["knowledge_base"] = load_folder_knowledge_base(
                folder_pdfs,
                folder_fingerprint,
                embedding_name,
                selected_chunk_size,
            )
            st.session_state["folder_index_key"] = folder_index_key
            st.session_state["knowledge_source"] = "folder"
            st.session_state["embedding_name"] = embedding_name
        except (RuntimeError, ValueError, OSError) as error:
            st.error(f"Could not load PDFs from {PDF_DIR}: {error}")

knowledge_base = st.session_state.get("knowledge_base")
if knowledge_base:
    if st.session_state.get("knowledge_source") == "folder":
        st.caption(
            f"Ready to chat: {len(folder_pdfs)} PDF(s) from "
            f"`{PDF_DIR.relative_to(PROJECT_DIR)}` · "
            f"{len(knowledge_base['chunks'])} indexed chunks"
        )
    model_names = []
    try:
        model_names = list_ollama_models()
    except RuntimeError as error:
        st.warning(str(error))

    default_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    if model_names:
        model_options = list(model_names)
        if default_model not in model_options:
            default_model = model_options[0]
        selected_model = st.selectbox(
            "Ollama chat model", model_options, index=model_options.index(default_model)
        )
    else:
        selected_model = st.text_input("Ollama chat model", value=default_model)

    for turn in st.session_state.get("conversation", []):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            with st.expander("Retrieved evidence"):
                if not turn["evidence"]:
                    st.info("No matching evidence was found.")
                for item in turn["evidence"]:
                    st.markdown(
                        "**{} — page {}** (score: {:.3f})".format(
                            item["source"], item["page"], item["score"]
                        )
                    )
                    st.write(item["text"])
                    st.divider()

    question = st.chat_input("Ask a question about the indexed PDFs")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            streamed_answer = False
            try:
                with st.spinner(
                    "Searching the lecture PDFs and generating your answer. "
                    "The local model may take a little while...",
                    show_time=True,
                ):
                    embedding_model = load_embedding_model(
                        st.session_state.get("embedding_name", embedding_name)
                    )
                    semantic_matches = knowledge_base["dense"].search(
                        question, embedding_model, top_k
                    )
                    if not has_relevant_evidence(semantic_matches):
                        best_score = (
                            semantic_matches[0]["score"]
                            if semantic_matches
                            else 0.0
                        )
                        retrieved = []
                        answer = (
                            "I couldn't find relevant evidence for that "
                            "question in the uploaded PDFs. The closest match "
                            f"had a similarity score of {best_score:.2f}, below "
                            f"the relevance threshold of "
                            f"{MIN_RELEVANCE_SCORE:.2f}. Try asking about "
                            "conversational AI ethics, LLM evaluation, SLMs, "
                            "or AI agents."
                        )
                    else:
                        if retrieval_method == "Semantic (FAISS)":
                            retrieved = semantic_matches
                        else:
                            retrieved = knowledge_base["keyword"].search(
                                question, top_k
                            )
                        answer = st.write_stream(
                            generate_answer_stream(
                                question,
                                retrieved,
                                selected_model,
                                max_tokens=96,
                            ),
                            cursor="▍",
                        )
                        streamed_answer = True
                if not streamed_answer:
                    st.markdown(answer)
                st.session_state.setdefault("conversation", []).append(
                    {
                        "question": question,
                        "answer": answer,
                        "evidence": retrieved,
                    }
                )
                with st.expander("Retrieved evidence"):
                    if not retrieved:
                        st.info("No matching evidence was found.")
                    for item in retrieved:
                        st.markdown(
                            "**{} — page {}** (score: {:.3f})".format(
                                item["source"],
                                item["page"],
                                item["score"],
                            )
                        )
                        st.write(item["text"])
                        st.divider()
            except (RuntimeError, ValueError, OSError) as error:
                st.error(f"Could not answer the question: {error}")
else:
    st.info(
        f"Add academic PDFs to `{PDF_DIR.relative_to(PROJECT_DIR)}` and "
        "the app will index them automatically."
    )

with st.expander("How to run the assignment experiments"):
    st.markdown(
        "Use `python experiments.py` with PDFs in `data/pdfs/` and an "
        "evaluation-question file. The runner compares 200/500/1000-token "
        "chunks, FAISS versus BM25, and no-context versus RAG answers. "
        "Human answer-quality ratings are intentionally not fabricated."
    )
