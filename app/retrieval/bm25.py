import re
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


BM25_TOP_K = 10


# These are rebuilt whenever a new document is ingested
_bm25_index = None
_bm25_documents: List[Document] = []


def tokenize(text: str) -> List[str]:
    """
    Convert text into simple lowercase tokens.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def create_bm25_index(
    chunks: List[Document],
) -> None:
    """
    Create a BM25 index from document chunks.
    """

    global _bm25_index
    global _bm25_documents

    if not chunks:
        raise ValueError(
            "Cannot create BM25 index from empty chunks."
        )

    _bm25_documents = chunks

    tokenized_documents = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    _bm25_index = BM25Okapi(
        tokenized_documents
    )


def bm25_search(
    query: str,
    top_k: int = BM25_TOP_K,
) -> List[Document]:
    """
    Search the BM25 index using keyword matching.
    """

    if _bm25_index is None:
        raise ValueError(
            "BM25 index has not been created. "
            "Please ingest a document first."
        )

    query_tokens = tokenize(query)

    scores = _bm25_index.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    return [
        _bm25_documents[index]
        for index in ranked_indexes
    ]