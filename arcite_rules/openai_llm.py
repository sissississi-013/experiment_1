"""
OpenAI LLM integration for rule generation.

Usage:
    from arcite_rules.openai_llm import make_openai_llm_fn
    from arcite_rules.rule_system import generate_rules

    llm = make_openai_llm_fn()
    rule_set = generate_rules("Find high-quality horse images", llm_call_fn=llm)
"""

import os


def make_openai_llm_fn(
    model: str = "gpt-4o",
    api_key: str | None = None,
    temperature: float = 0.0,
):
    """
    Returns a function(system_prompt, user_prompt) -> str backed by OpenAI.

    Args:
        model:       OpenAI model ID. gpt-4o is recommended for rule generation.
        api_key:     OpenAI API key. Defaults to OPENAI_API_KEY env var.
        temperature: 0.0 = deterministic (best for structured JSON output).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required: pip install openai")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=key)

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    return call
