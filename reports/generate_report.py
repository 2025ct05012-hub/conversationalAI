"""Generate the four-page assignment report PDF."""

import csv
from html import escape
from pathlib import Path
import textwrap

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path(__file__).with_name("Academic_PDF_RAG_Project_Report.pdf")
FLOW_IMAGE = Path(__file__).with_name("flow.jpeg")
CHATBOT_IMAGE = Path(__file__).with_name("Chatbot.png")
RESULTS_DIR = OUTPUT.parent.parent / "results"


def _read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _percent(value):
    return f"{float(value) * 100:.1f}%" if value else "N/A"


def _score(value):
    return f"{float(value):.3f}" if value else "N/A"


def _excerpt(text, maximum=220):
    if len(text) <= maximum:
        return escape(text).replace("\n", "<br/>")
    return escape(text[:maximum].rsplit(" ", 1)[0] + "...").replace(
        "\n", "<br/>"
    )


def _draw_workflow() -> None:
    width, height = 1800, 980
    image = PILImage.new("RGB", (width, height), "#f5f8fc")
    draw = ImageDraw.Draw(image)
    regular_path = Path(r"C:\Windows\Fonts\arial.ttf")
    bold_path = Path(r"C:\Windows\Fonts\arialbd.ttf")

    def font(size, bold=False):
        path = bold_path if bold else regular_path
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            return ImageFont.load_default()

    title_font = font(42, bold=True)
    section_font = font(25, bold=True)
    node_title_font = font(22, bold=True)
    node_body_font = font(17)
    label_font = font(16, bold=True)

    navy = "#183153"
    blue = "#e7f0fb"
    blue_border = "#3e78a8"
    green = "#e6f4ee"
    green_border = "#36825b"
    orange = "#fff2df"
    orange_border = "#c17a22"
    gray = "#435466"

    def center_text(text, y, selected_font, fill=navy):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=selected_font)
        draw.text(
            ((width - (right - left)) / 2, y),
            text,
            font=selected_font,
            fill=fill,
        )

    center_text("Academic PDF RAG — System Workflow", 28, title_font)
    draw.rounded_rectangle(
        (42, 105, width - 42, 426),
        radius=24,
        fill="#edf3f9",
        outline="#cad7e5",
        width=2,
    )
    draw.text((68, 120), "1. DOCUMENT INGESTION (ONCE PER PDF SET)", font=section_font, fill=navy)

    def node(box, title, body, fill, border):
        left, top, right, bottom = box
        draw.rounded_rectangle(
            box, radius=18, fill=fill, outline=border, width=3
        )
        title_lines = textwrap.wrap(title, width=23)
        title_y = top + 16
        for line in title_lines:
            draw.text((left + 14, title_y), line, font=node_title_font, fill=navy)
            title_y += 27
        body_lines = textwrap.wrap(body, width=30)
        body_y = max(title_y + 5, top + 59)
        for line in body_lines[:3]:
            draw.text((left + 14, body_y), line, font=node_body_font, fill=gray)
            body_y += 22

    row1 = [
        (65, 183, 365, 365),
        (400, 183, 700, 365),
        (735, 183, 1035, 365),
        (1070, 183, 1370, 365),
        (1405, 183, 1735, 365),
    ]
    node(row1[0], "Academic PDFs", "PDF files in data/pdfs/", blue, blue_border)
    node(row1[1], "Extract text", "PyMuPDF; Tesseract OCR for scanned pages", blue, blue_border)
    node(row1[2], "Chunk pages", "Overlapping chunks retain source and page", blue, blue_border)
    node(row1[3], "Create vectors", "Sentence Transformer embeddings", blue, blue_border)
    node(
        row1[4],
        "Search indexes",
        "FAISS semantic + BM25 keyword search",
        green,
        green_border,
    )

    def arrow(start, end, color=blue_border, label=None, label_pos=None):
        draw.line((start, end), fill=color, width=5)
        import math

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        length = 17
        left = (
            end[0] - length * math.cos(angle - math.pi / 6),
            end[1] - length * math.sin(angle - math.pi / 6),
        )
        right = (
            end[0] - length * math.cos(angle + math.pi / 6),
            end[1] - length * math.sin(angle + math.pi / 6),
        )
        draw.polygon([end, left, right], fill=color)
        if label and label_pos:
            draw.text(label_pos, label, font=label_font, fill=color)

    for index in range(4):
        arrow(
            (row1[index][2] + 8, 274),
            (row1[index + 1][0] - 8, 274),
        )

    draw.rounded_rectangle(
        (42, 458, width - 42, 930),
        radius=24,
        fill="#f1f6f2",
        outline="#cadfd2",
        width=2,
    )
    draw.text((68, 473), "2. QUESTION ANSWERING (PER QUESTION)", font=section_font, fill=navy)

    row2 = [
        (65, 555, 312, 737),
        (356, 555, 603, 737),
        (647, 555, 894, 737),
        (938, 555, 1185, 737),
        (1229, 555, 1476, 737),
        (1520, 555, 1767, 737),
    ]
    node(row2[0], "User question", "Question entered in chatbot", blue, blue_border)
    node(row2[1], "Embed query", "Use the same embedding model", blue, blue_border)
    node(row2[2], "Retrieve top-k", "Search FAISS or BM25 for passages", green, green_border)
    node(row2[3], "Relevance gate", "Semantic score must be at least 0.30", orange, orange_border)
    node(row2[4], "Local Ollama", "Answer using retrieved context only", blue, blue_border)
    node(row2[5], "Cited answer", "Show answer and PDF page references", green, green_border)
    for index in range(5):
        color = orange_border if index == 2 else blue_border
        arrow(
            (row2[index][2] + 8, 646),
            (row2[index + 1][0] - 8, 646),
            color=color,
        )

    draw.rounded_rectangle(
        (948, 794, 1478, 893),
        radius=15,
        fill="#fff2df",
        outline=orange_border,
        width=2,
    )
    draw.text((970, 805), "Below threshold: abstain", font=node_title_font, fill=navy)
    draw.text(
        (970, 844),
        "Tell the user no relevant PDF evidence was found.",
        font=node_body_font,
        fill=gray,
    )
    draw.line((1060, 737, 1060, 790), fill=orange_border, width=4)
    draw.polygon([(1060, 790), (1051, 775), (1069, 775)], fill=orange_border)
    image.save(FLOW_IMAGE, optimize=True)


def build_report() -> None:
    _draw_workflow()
    with PILImage.open(CHATBOT_IMAGE) as chatbot_screenshot:
        chatbot_width = 172 * mm
        chatbot_height = (
            chatbot_width
            * chatbot_screenshot.height
            / chatbot_screenshot.width
        )
    retrieval_summary = _read_csv("retrieval_summary.csv")
    answer_comparisons = _read_csv("answer_comparison.csv")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183153"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#24527a"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableBody",
            parent=styles["BodyText"],
            fontSize=7,
            leading=8.5,
            spaceAfter=0,
        )
    )

    def p(text, style="SmallBody"):
        return Paragraph(text, styles[style])

    def table_style(font_size=7):
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa9b8")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fa")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="Design and Implementation of a RAG-Based Academic PDF QA System",
        author="Experiential Learning Project",
    )
    story = [
        Spacer(1, 24 * mm),
        p("EXPERIENTIAL LEARNING PROJECT REPORT", "CoverTitle"),
        Spacer(1, 8 * mm),
        p(
            "Design and Implementation of a RAG-Based Academic PDF "
            "Question Answering System Using Python",
            "CoverTitle",
        ),
        Spacer(1, 8 * mm),
        p(
            "A local application that retrieves page-aware evidence from "
            "academic PDFs before generating an answer with a large language model.",
            "BodyText",
        ),
        Spacer(1, 12 * mm),
        p("<b>Technology:</b> Python 3.10+, PyMuPDF, Tesseract OCR, "
          "Sentence Transformers, FAISS, Ollama, Streamlit and Pandas"),
        p("<b>Experiment environment:</b> Local Python / Visual Studio Code"),
        p("<b>Group:</b> 4"),
        p("<b>Student(s):</b> 2025ct05012, 2025ct05018, 2025ct05006, 2025ct05004,2025ct05026 "),
        Spacer(1, 8 * mm),
        p("Abstract", "Section"),
        p(
            "This project implements a retrieval-augmented generation (RAG) "
            "workflow for question answering over academic PDF documents. "
            "PDF text is extracted with page metadata (using local Tesseract "
            "OCR for scanned pages), divided into overlapping "
            "embedding-token chunks, encoded using a sentence-transformer, and "
            "indexed with FAISS cosine-similarity search. A BM25 keyword retriever "
            "provides a comparison method. Retrieved evidence is supplied to a "
            "locally hosted Ollama model, which is instructed to answer only from "
            "that evidence and cite source pages, streaming partial answers as "
            "they are generated. Weak semantic matches below a 0.30 similarity "
            "threshold are rejected before generation. The accompanying experiment "
            "runner compares chunk sizes, retrievers, and context/no-context "
            "answering. Corpus-specific measurements must be produced using "
            "authorized course PDFs; they are not fabricated in this report.",
        ),
        PageBreak(),
        p("1. Introduction and System Design", "Section"),
        p(
            "General-purpose language models can provide plausible but "
            "unsupported answers and do not automatically know the contents of "
            "a particular lecture PDF. RAG addresses this limitation by "
            "retrieving relevant passages at question time and conditioning "
            "generation on those passages. A relevance gate declines weak "
            "matches, and accepted answers stream from the local model. The "
            "goal is to support verifiable "
            "academic question answering, not to replace source inspection.",
        ),
        p("Objectives", "Section"),
        p(
            "1. Extract text while preserving PDF filename and one-based page "
            "number.<br/>2. Split page text into overlapping chunks using the "
            "embedding tokenizer.<br/>3. Encode and index chunks in FAISS; "
            "implement BM25 as a lexical baseline.<br/>4. Retrieve evidence, "
            "generate a grounded answer through local Ollama, and show source "
            "pages.<br/>5. Compare chunk sizes, retrieval methods, and "
            "context/no-context answers with reproducible outputs.",
        ),
        p("Architecture", "Section"),
        p(
            "<b>Ingestion:</b> uploaded PDF → PyMuPDF page extraction → "
            "tokenizer-based chunking with overlap → sentence-transformer "
            "embeddings → normalized FAISS inner-product index. "
            "<b>Question:</b> user query → semantic FAISS or BM25 retrieval → "
            "evidence with source/page metadata → relevance check → Ollama prompt "
            "→ streamed cited response. The app declines weak semantic matches "
            "below a 0.30 cosine-similarity threshold rather than treating an "
            "arbitrary nearest chunk as evidence.",
        ),
        Spacer(1, 3 * mm),
        Image(
            str(FLOW_IMAGE),
            width=172 * mm,
            height=172 * mm * 980 / 1800,
            hAlign="CENTER",
        ),
        p(
            "Figure 1. End-to-end local PDF ingestion, retrieval, relevance "
            "filtering, and cited-answer workflow.",
            "TableBody",
        ),
        p("Implementation and technology", "Section"),
        p(
            "The Streamlit interface automatically indexes PDFs in the project "
            "folder, allows optional multi-PDF upload, chunk-size "
            "selection, retrieval-method selection, and evidence inspection. "
            "PyMuPDF extracts selectable text page by page and renders scanned "
            "pages for local Tesseract OCR. Sentence Transformers supplies "
            "the model tokenizer and vectors; long chunks are split into "
            "model-sized windows and their embeddings averaged so that "
            "large-chunk experiments do not silently discard trailing text. "
            "FAISS IndexFlatIP searches L2-normalized vectors, equivalent to "
            "cosine ranking. The BM25 "
            "implementation uses term frequency, document frequency, and "
            "document-length normalization. Ollama's local HTTP API performs "
            "chat completion. Pandas writes experiment tables to CSV.",
        ),
        p("Privacy and use", "Section"),
        p(
            "PDF content and prompts are sent to the configured local Ollama "
            "service, not a hosted LLM API. The embedding model is downloaded "
            "on first use. Only documents the user is authorized to process "
            "should be included; course materials are not bundled with this project.",
        ),
        PageBreak(),
        p("2. Experimental Methodology", "Section"),
        p(
            "Use the same evaluation questions and PDF corpus for each condition. "
            "Record the corpus filenames, embedding model, Ollama model, top-k, "
            "software versions, and date in the final submission. Results depend "
            "on the selected course corpus and models.",
        ),
        p("Experiment 1 — chunk size", "Section"),
        p(
            "Compare target sizes of 200, 500, and 1000 embedding-model tokens "
            "(60-token overlap, reduced for the smallest chunk if necessary). "
            "Keep the embedding model, corpus, query set, and top-k fixed. "
            "Record retrieval hit rate, precision@k, reciprocal rank, and "
            "retrieval latency. Smaller chunks may be more precise; larger "
            "chunks may preserve more context. This is a hypothesis, not a result.",
        ),
        p("Experiment 2 — retrieval method", "Section"),
        p(
            "Compare normalized-vector FAISS semantic retrieval with BM25 "
            "keyword retrieval on identical questions and chunks. For each "
            "question, label expected source PDFs and/or pages in "
            "<i>data/evaluation_questions.json</i>. Hit rate@k is the fraction "
            "of labeled questions with at least one matching returned result; "
            "precision@k is matching results divided by k; reciprocal rank is "
            "the inverse rank of the first matching result. Unlabeled questions "
            "remain blank rather than being scored as failures.",
        ),
        p("Experiment 3 — RAG versus normal LLM", "Section"),
        p(
            "Ask the same questions of the same Ollama model with no retrieved "
            "context and with FAISS evidence, fixing the RAG chunk size at 500 "
            "tokens. The no-context condition is "
            "answered from the model's pretrained knowledge without any "
            "document context. Review each pair against the PDFs and rate correctness, "
            "relevance, and source grounding from 1 (poor) to 5 (excellent). "
            "Record reviewer notes; do not infer quality from answer fluency.",
        ),
        p("Reproducibility", "Section"),
        p(
            "Place authorized PDFs in <i>data/pdfs/</i>, copy the example "
            "question JSON to <i>data/evaluation_questions.json</i>, add "
            "expected pages/sources, and run <i>python experiments.py</i>. "
            "The runner exports retrieval_metrics.csv and answer_comparison.csv. "
            "Repeat runs with the same model/configuration and note model "
            "versions. Latency is hardware-dependent.",
        ),
        Spacer(1, 3 * mm),
        p("Chatbot interface", "Section"),
        Image(
            str(CHATBOT_IMAGE),
            width=chatbot_width,
            height=chatbot_height,
            hAlign="CENTER",
        ),
        p(
            "Figure 2. Streamlit chatbot with the lecture PDFs indexed and a "
            "source-grounded question-and-answer interaction.",
            "TableBody",
        ),
        PageBreak(),
        p("3. Results, Discussion and Conclusion", "Section"),
        p(
            "The retrieval experiment was run on the three lecture PDFs using "
            "nine source-labeled questions and top-k = 3. The table below is "
            "loaded from <i>results/retrieval_summary.csv</i>; latency is the "
            "mean retrieval time per query on this local machine.",
        ),
    ]

    if retrieval_summary:
        retrieval_table_data = [
            [
                p("Chunk tokens", "TableBody"),
                p("Retriever", "TableBody"),
                p("Hit@3", "TableBody"),
                p("P@3", "TableBody"),
                p("MRR", "TableBody"),
                p("Latency", "TableBody"),
            ]
        ]
        for row in retrieval_summary:
            retrieval_table_data.append(
                [
                    p(escape(row["chunk_size_tokens"]), "TableBody"),
                    p(escape(row["method"].replace("_", " ")), "TableBody"),
                    p(_percent(row["hit_rate_at_k"]), "TableBody"),
                    p(_percent(row["precision_at_k"]), "TableBody"),
                    p(_score(row["mean_reciprocal_rank"]), "TableBody"),
                    p(
                        f"{float(row['mean_latency_seconds']) * 1000:.1f} ms",
                        "TableBody",
                    ),
                ]
            )
        results_table = Table(
            retrieval_table_data,
            colWidths=[24 * mm, 37 * mm, 19 * mm, 19 * mm, 18 * mm, 20 * mm],
            repeatRows=1,
        )
        results_table.setStyle(table_style())
    else:
        results_table = p(
            "Retrieval metrics have not been generated. Run "
            "<i>python experiments.py</i> to populate the results table."
        )

    answer_sections = []
    if answer_comparisons:
        for row in answer_comparisons:
            answer_sections.extend(
                [
                    p(
                        "<b>Question:</b> "
                        + escape(row["question"]),
                        "TableBody",
                    ),
                    p(
                        "<b>Normal LLM:</b> "
                        + _excerpt(row["baseline_answer"]),
                        "TableBody",
                    ),
                    p(
                        "<b>RAG (500-token chunks):</b> "
                        + _excerpt(row["rag_answer"]),
                        "TableBody",
                    ),
                    p(
                        "<b>Retrieved source pages:</b> "
                        + _excerpt(row["rag_sources"], 180),
                        "TableBody",
                    ),
                    Spacer(1, 2 * mm),
                ]
            )
    else:
        answer_sections.append(
            p(
                "No RAG-versus-normal-LLM answer pairs have been generated "
                "yet. These outputs and human scores will appear after the "
                "experiment runner completes.",
                "TableBody",
            )
        )

    score_rows = []
    for row in answer_comparisons:
        score_rows.append(
            [
                p(escape(row["question"]), "TableBody"),
                p(
                    escape(row["baseline_correctness_1_to_5"] or "Pending"),
                    "TableBody",
                ),
                p(
                    escape(row["rag_correctness_1_to_5"] or "Pending"),
                    "TableBody",
                ),
                p(
                    escape(row["baseline_relevance_1_to_5"] or "Pending"),
                    "TableBody",
                ),
                p(
                    escape(row["rag_relevance_1_to_5"] or "Pending"),
                    "TableBody",
                ),
                p(
                    escape(row["baseline_grounding_1_to_5"] or "Pending"),
                    "TableBody",
                ),
                p(
                    escape(row["rag_grounding_1_to_5"] or "Pending"),
                    "TableBody",
                ),
            ]
        )
    if not score_rows:
        score_rows = [
            [
                p("No answer pair", "TableBody"),
                *[p("Pending", "TableBody") for _ in range(6)],
            ]
        ]
    answer_score_table = Table(
        [
            [
                p("Question", "TableBody"),
                p("Base C", "TableBody"),
                p("RAG C", "TableBody"),
                p("Base R", "TableBody"),
                p("RAG R", "TableBody"),
                p("Base G", "TableBody"),
                p("RAG G", "TableBody"),
            ],
            *score_rows,
        ],
        colWidths=[53 * mm, 13 * mm, 13 * mm, 13 * mm, 13 * mm, 13 * mm, 13 * mm],
        repeatRows=1,
    )
    answer_score_table.setStyle(table_style())

    story.extend(
        [
            Spacer(1, 3 * mm),
            results_table,
            p(
                "All nine labeled questions achieved 100% hit rate. Semantic "
                "retrieval had 80.0% precision@3 and MRR 1.000 at each chunk "
                "size; BM25 had 64.4–66.7% precision@3 and MRR 0.926. BM25 "
                "was faster in this run, while semantic search ranked the "
                "expected source first for every question.",
                "TableBody",
            ),
            p("RAG versus normal LLM", "Section"),
            p(
                "The experiment generated the following answer pair. The "
                "normal model used no document context; the RAG answer used "
                "retrieved passages. Excerpts are included below and the full "
                "outputs remain in <i>results/answer_comparison.csv</i>.",
                "TableBody",
            ),
            *answer_sections,
            p(
                "Output limitation: the saved RAG answer for this sample is "
                "truncated mid-citation and includes retrieved chunks from "
                "another lecture. Review the full answer and source relevance "
                "before assigning its human quality scores.",
                "TableBody",
            ),
            p(
                "Human ratings (1–5): C = correctness, R = relevance, "
                "G = grounding. These cells remain pending because the answers "
                "must be checked against the lecture PDFs by the project team; "
                "automatically assigning quality scores would not be a valid "
                "human evaluation.",
                "TableBody",
            ),
            answer_score_table,
            p("Discussion", "Section"),
            p(
                "Results show that semantic search ranked relevant lecture "
                "sources effectively for this small labeled set. OCR-derived "
                "text and generated answers still require source verification. "
                "Retrieval metrics measure matching source files, not the "
                "factual correctness of generated answers.",
                "TableBody",
            ),
            p("Conclusion", "Section"),
            p(
                "The local system demonstrates PDF OCR, chunking, embeddings, "
                "FAISS/BM25 retrieval, and source-cited Ollama answers. It "
                "performs well on the labeled retrieval questions in this run; "
                "human answer-quality ratings remain to be completed.",
                "TableBody",
            ),
        ]
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(19 * mm, 10 * mm, "Academic PDF RAG Project")
        canvas.drawRightString(A4[0] - 19 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    build_report()
