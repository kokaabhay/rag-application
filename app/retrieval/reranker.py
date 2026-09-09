import json
import requests
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings


RERANK_TOP_K = 3

RERANK_URL = "https://openrouter.ai/api/v1/rerank"


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = RERANK_TOP_K,
) -> List[Tuple[Document, float]]:

    if not documents:
        return []

    try:
        response = requests.post(
            verify=False,
            url=RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
                "query": query,
                "documents": [
                    {
                        "text": document.page_content
                    }
                    for document in documents
                ],
                "top_n": top_k,
            }),
        )

        response.raise_for_status()

        results = response.json()

        reranked_documents = []

        for result in results["results"]:
            index = result["index"]
            score = float(result["relevance_score"])

            document = documents[index]

            reranked_documents.append(
                (document, score)
            )

        return reranked_documents

    except Exception as e:
        print("Calling reranking may have failed:", str(e))
        return []