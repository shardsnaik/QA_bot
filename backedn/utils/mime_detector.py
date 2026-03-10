"""
MIME type detection utility.
Uses python-magic to sniff magic bytes, falling back to the client-supplied content type.
"""

from __future__ import annotations


def detect_mime_type(content: bytes, fallback: str | None = None) -> str:
    """
    Detect the MIME type of *content* using magic bytes.

    Falls back to *fallback* (typically the HTTP Content-Type header)
    if python-magic is unavailable or cannot determine the type.

    Parameters
    ----------
    content : bytes
        Raw file content.
    fallback : str, optional
        Client-supplied MIME type to use as fallback.

    Returns
    -------
    str
        Detected MIME type, e.g. ``"text/plain"``.
    """
    try:
        import magic  # python-magic-bin on Windows
        mime = magic.from_buffer(content, mime=True)
        if mime:
            return mime
    except ImportError:
        pass
    except Exception:
        pass

    return fallback or "application/octet-stream"
