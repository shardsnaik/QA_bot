"""
PDF text extraction utility using pypdf.
Extracts raw text from PDF bytes.
"""

from __future__ import annotations

import io
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract all text from a PDF file's bytes.

    Parameters
    ----------
    content : bytes
        The raw bytes of the PDF file.

    Returns
    -------
    str
        The extracted text from all pages.
    """
    try:
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        
        extracted = "\n\n".join(full_text)
        logger.info("Extracted %d characters from PDF", len(extracted))
        return extracted
        
    except Exception as e:
        logger.error("Failed to extract text from PDF: %s", e)
        raise ValueError(f"Could not extract text from PDF: {e}")
