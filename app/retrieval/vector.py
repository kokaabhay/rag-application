from typing import List
import app.core.huggingface
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata


CHROMA_DIRECTORY = "chroma_db"
COLLECTION_NAME = "customer_support"
VECTOR_TOP_K = 10


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(chunks: List[Document]) -> Chroma:
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIRECTORY,
    )

    existing_data = vector_store.get()

    if existing_data["ids"]:
        vector_store.delete(ids=existing_data["ids"])

    vector_store.add_documents(chunks)

    return vector_store


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIRECTORY,
    )


def vector_search(
    query: str,
    top_k: int = VECTOR_TOP_K,
) -> List[Document]:

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query,
        k=top_k,
    )