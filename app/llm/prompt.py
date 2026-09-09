def build_prompt(
    query: str,
    context: str,
) -> str:
    """
    Build the prompt sent to the LLM.
    """

    return f"""
You are a customer support assistant.

Answer the user's question using ONLY the information
provided in the knowledge base context.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- If the context does not contain enough information,
  say that the information is not available.
- Give a concise and helpful answer.
- Do not mention the retrieval process.
- Do not include a Sources section.

Knowledge Base Context:
------------------------
{context}
------------------------

User Question:
{query}

Answer:
""".strip()