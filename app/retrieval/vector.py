from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "customer_support"

VECTOR_TOP_K = 10


# Load embedding model once when the application starts
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(
    chunks: List[Document],
) -> Chroma:
    """
    Create a Chroma vector store from document chunks.

    Existing vectors are deleted because this application
    supports only one knowledge-base document.
    """

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIRECTORY,
    )

    # Remove previous knowledge-base vectors
    existing_data = vector_store.get()

    if existing_data["ids"]:
        vector_store.delete(
            ids=existing_data["ids"]
        )

    # Add the new chunks
    vector_store.add_documents(chunks)

    return vector_store


def get_vector_store() -> Chroma:
    """
    Return the existing Chroma vector store.
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIRECTORY,
    )


def vector_search(
    query: str,
    top_k: int = VECTOR_TOP_K,
) -> List[Document]:
    """
    Search Chroma using semantic similarity.
    """

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query,
        k=top_k,
    )

    return results