from pathlib import Path
import shutil

from langchain_core.documents import Document

from app.ingestion.loader import load_document
from app.ingestion.chunker import split_documents


DOCUMENT_DIRECTORY = Path("data/documents")
STANDARD_FILENAME = "knowledge_base"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".md",
}


def save_uploaded_document(
    source_path: str,
    extension: str,
) -> Path:
    """
    Save the uploaded document using the standard
    knowledge-base filename.

    Any previous knowledge-base document is removed first.
    """

    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Uploaded document does not exist: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Uploaded path is not a file: {source}"
        )

    extension = extension.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    DOCUMENT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove the previous knowledge-base document
    for old_file in DOCUMENT_DIRECTORY.glob(
        f"{STANDARD_FILENAME}.*"
    ):
        old_file.unlink()

    destination = (
        DOCUMENT_DIRECTORY
        / f"{STANDARD_FILENAME}{extension}"
    )

    shutil.copy2(source, destination)

    return destination


def ingest_document(
    source_path: str,
    extension: str,
) -> list[Document]:
    """
    Complete document ingestion preprocessing:

    1. Save uploaded document
    2. Load using Unstructured
    3. Split into chunks
    """

    saved_path = save_uploaded_document(
        source_path,
        extension,
    )

    documents = load_document(saved_path)

    chunks = split_documents(documents)

    return chunks