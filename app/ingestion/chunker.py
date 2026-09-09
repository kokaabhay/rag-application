from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunking configuration
#500 means 500 characters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Split documents into smaller chunks for retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "Document splitting produced no chunks."
        )

    return chunks