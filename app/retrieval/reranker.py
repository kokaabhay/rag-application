from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANK_TOP_K = 3
RERANK_SCORE_THRESHOLD = 0.5


# Load the reranker once when the application starts
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = RERANK_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Rerank retrieved documents using a CrossEncoder.

    Returns:
        List of (Document, score) tuples.
    """

    if not documents:
        return []

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = reranker_model.predict(pairs)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked_results[:top_k]