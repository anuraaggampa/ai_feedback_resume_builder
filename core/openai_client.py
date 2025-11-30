import os
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
DEFAULT_MODEL = "gpt-4.1-mini"

# 🔹 Global usage log for the current run
_USAGE_LOG: List[Dict[str, Any]] = []

# 🔹 Pricing per 1M tokens (USD) — from OpenAI pricing page
# https://openai.com/api/pricing
MODEL_PRICES = {
    "gpt-4.1-mini": {  # your DEFAULT_MODEL
        "input_per_m": 0.25,   # $0.25 / 1M input tokens :contentReference[oaicite:0]{index=0}
        "output_per_m": 2.00,  # $2.00 / 1M output tokens :contentReference[oaicite:1]{index=1}
    },
    "gpt-4.1": {
        "input_per_m": 2.00,   # $2.00 / 1M input :contentReference[oaicite:2]{index=2}
        "output_per_m": 8.00,  # $8.00 / 1M output :contentReference[oaicite:3]{index=3}
    },
    # add others if you use them
}

def reset_usage_log() -> None:
    """
    Clear the accumulated token usage.
    Call this once before starting a 'generate resume' run.
    """
    _USAGE_LOG.clear()


def _log_usage(model: str, usage: Optional[Any]) -> None:
    """
    Store usage info from a single OpenAI call into _USAGE_LOG.
    """
    if usage is None:
        return
    try:
        _USAGE_LOG.append(
            {
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        )
    except Exception:
        # Be defensive: never break the app because of logging
        pass


def get_usage_summary_with_cost() -> Dict[str, Any]:
    """
    Aggregate token usage across all logged calls and estimate USD cost.
    Returns something like:
    {
      "total_prompt_tokens": int,
      "total_completion_tokens": int,
      "total_tokens": int,
      "per_model": { model_name: {...} },
      "total_cost_usd": float,
    }
    """
    summary: Dict[str, Any] = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "per_model": {},
        "total_cost_usd": 0.0,
    }

    for entry in _USAGE_LOG:
        model = entry.get("model", DEFAULT_MODEL)
        pt = entry.get("prompt_tokens", 0)
        ct = entry.get("completion_tokens", 0)
        tt = entry.get("total_tokens", 0)

        summary["total_prompt_tokens"] += pt
        summary["total_completion_tokens"] += ct
        summary["total_tokens"] += tt

        if model not in summary["per_model"]:
            summary["per_model"][model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }

        summary["per_model"][model]["prompt_tokens"] += pt
        summary["per_model"][model]["completion_tokens"] += ct
        summary["per_model"][model]["total_tokens"] += tt

    # 🔹 Compute cost per model
    for model, data in summary["per_model"].items():
        prices = MODEL_PRICES.get(model)
        if not prices:
            continue

        in_price = prices["input_per_m"]
        out_price = prices["output_per_m"]

        in_cost = (data["prompt_tokens"] / 1_000_000) * in_price
        out_cost = (data["completion_tokens"] / 1_000_000) * out_price
        total_cost = in_cost + out_cost

        data["cost_usd"] = total_cost
        summary["total_cost_usd"] += total_cost

    return summary



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

    # 🔹 Log usage for cost tracking (doesn't change return type)
    try:
        _log_usage(model, getattr(resp, "usage", None))
    except Exception:
        pass

    return resp.choices[0].message.content.strip()

