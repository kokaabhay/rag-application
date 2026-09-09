from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".md",
}


def load_document(file_path: Path) -> List[Document]:
    """
    Load a document using Unstructured and convert it
    into LangChain Document objects.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    loader = UnstructuredLoader(str(file_path))

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No content could be extracted from the document."
        )

    return documents