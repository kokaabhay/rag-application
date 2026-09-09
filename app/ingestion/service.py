from pathlib import Path
import shutil
from app.retrieval.bm25 import create_bm25_index
from langchain_core.documents import Document

from app.ingestion.loader import load_document
from app.ingestion.chunker import split_documents
from app.retrieval.vector import create_vector_store


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

    Any previous knowledge-base document is removed.
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

    # Delete previous knowledge-base document
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
    Complete ingestion pipeline:

    1. Save uploaded document
    2. Load document
    3. Split into chunks
    4. Create vector index
    """

    saved_path = save_uploaded_document(
        source_path,
        extension,
    )

    documents = load_document(saved_path)

    chunks = split_documents(documents)

    # Create/update Chroma vector index
    create_vector_store(chunks)
    create_bm25_index(chunks)
    # Create/update BM25 keyword index
    return chunks