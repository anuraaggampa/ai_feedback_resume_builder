import tempfile
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


def _get_docling_converter():
    if not DOCLING_AVAILABLE:
        return None
    return DocumentConverter()


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Try Docling first. If it fails, fallback to a simple PyPDF2 text extractor.
    Keep user messages clean (no low-level error dumps).
    """
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = uploaded_file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)

    text = ""

    # 1) Docling
    converter = _get_docling_converter()
    if converter is not None:
        try:
            result = converter.convert(tmp_path)
            text = result.document.export_to_markdown() or ""
        except Exception:
            # Silent fallback, small info only
            st.info("Using an alternate extractor for this PDF.")
            text = ""

    # 2) PyPDF2 fallback
    if not text:
        try:
            with open(tmp_path, "rb") as f:
                reader = PdfReader(f)
                pages_text = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                text = "\n\n".join(pages_text).strip()
        except Exception:
            st.error("We couldn’t read text from this PDF. Please paste the text manually.")
            text = ""

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    return text
