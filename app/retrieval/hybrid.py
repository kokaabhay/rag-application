from typing import List, Dict

from langchain_core.documents import Document

from app.retrieval.vector import vector_search
from app.retrieval.bm25 import bm25_search


HYBRID_TOP_K = 10
RRF_K = 60


def hybrid_search(
    query: str,
    top_k: int = HYBRID_TOP_K,
) -> List[Document]:
    """
    Perform vector + BM25 retrieval and combine
    the rankings using Reciprocal Rank Fusion.
    """

    vector_results = vector_search(
        query,
        top_k=top_k,
    )

    bm25_results = bm25_search(
        query,
        top_k=top_k,
    )

    scores: Dict[str, float] = {}
    documents: Dict[str, Document] = {}

    # Process vector ranking
    for rank, document in enumerate(
        vector_results,
        start=1,
    ):
        key = document.page_content

        scores[key] = scores.get(key, 0) + (
            1 / (RRF_K + rank)
        )

        documents[key] = document

    # Process BM25 ranking
    for rank, document in enumerate(
        bm25_results,
        start=1,
    ):
        key = document.page_content

        scores[key] = scores.get(key, 0) + (
            1 / (RRF_K + rank)
        )

        documents[key] = document

    # Sort by combined RRF score
    ranked_documents = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        documents[key]
        for key, _ in ranked_documents[:top_k]
    ]