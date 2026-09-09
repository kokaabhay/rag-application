
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank_documents
from app.llm.prompt import build_prompt
from app.llm.service import generate_answer


FALLBACK_MESSAGE = (
    "I'm sorry, but I couldn't find relevant information "
    "in the knowledge base to answer your question."
)


def chat(query: str) -> str:
    """
    Complete RAG pipeline.

    Query
        ↓
    Hybrid Search
        ↓
    OpenRouter Reranking
        ↓
    Relevance Check
        ↓
    Prompt
        ↓
    LLM
        ↓
    Answer
    """

    # 1. Retrieve candidate documents
    documents = hybrid_search(query)

    if not documents:
        return FALLBACK_MESSAGE

    # 2. Rerank the candidates
    reranked_documents = rerank_documents(
        query,
        documents,
    )

    if not reranked_documents:
        return FALLBACK_MESSAGE

    # 3. Get the best relevance score
    best_score = reranked_documents[0][1]

    # 4. Build context from top-ranked documents
    context = "\n\n".join(
        document.page_content
        for document, _ in reranked_documents
    )

    # 5. Build LLM prompt
    prompt = build_prompt(
        query=query,
        context=context,
    )

    # 6. Generate answer
    answer = generate_answer(prompt)

    return answer

