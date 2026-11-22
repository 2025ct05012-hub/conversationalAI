"""Page-aware text extraction from PDF files."""

from io import BytesIO
import os
from pathlib import Path
import shutil
from typing import Dict, List

import pymupdf
import pytesseract
from PIL import Image


def _configure_tesseract() -> None:
    configured_path = os.environ.get("TESSERACT_CMD")
    discovered_path = shutil.which("tesseract")
    default_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    executable = configured_path or discovered_path
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    elif default_path.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(default_path)


def extract_pdf_pages(
    pdf_path: str, ocr_language: str = "eng"
) -> List[Dict[str, object]]:
    """Extract text, using local OCR for image-only pages."""
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    _configure_tesseract()
    pages: List[Dict[str, object]] = []
    with pymupdf.open(str(path)) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                try:
                    image_bytes = page.get_pixmap(dpi=200, alpha=False).tobytes(
                        "png"
                    )
                    with Image.open(BytesIO(image_bytes)) as image:
                        text = pytesseract.image_to_string(
                            image, lang=ocr_language, timeout=60
                        ).strip()
                except pytesseract.TesseractNotFoundError as error:
                    raise RuntimeError(
                        "This PDF contains scanned pages, but the Tesseract "
                        "OCR engine was not found. Install Tesseract OCR and "
                        "restart the app."
                    ) from error
            if text:
                pages.append(
                    {"source": path.name, "page": page_index, "text": text}
                )
    if not pages:
        raise ValueError(
            f"No extractable text found in {path.name}; it may be scanned "
            "and require OCR."
        )
    return pages
