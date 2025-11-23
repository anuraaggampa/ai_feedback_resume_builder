import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
DEFAULT_MODEL = "gpt-4.1-mini"


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in environment variables.")
    return OpenAI(api_key=api_key)


def chat_completion(
    system_msg: str,
    user_msg: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    Small wrapper around OpenAI chat.completions.create
    Returns the text content only.
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
