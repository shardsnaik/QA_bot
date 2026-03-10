"""
Text chunking utility using LangChain's RecursiveCharacterTextSplitter.
Splits long text into smaller overlapping chunks for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str) -> list[str]:
    """
    Split *text* into overlapping chunks.

    Returns
    -------
    list[str]
        A list of text chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
