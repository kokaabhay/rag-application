import httpx

from app.config import settings


def generate_answer(
    prompt: str,
) -> str:
    """
    Send the prompt to the configured LLM
    and return the generated answer.
    """

    url = f"{settings.llm_base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    response = httpx.post(
        url,
        headers=headers,
        json=payload,
        timeout=60.0,
        verify=False,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]