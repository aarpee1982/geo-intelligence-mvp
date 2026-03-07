"""
DeepSeek API integration module (placeholder, not yet wired).

HOW TO ACTIVATE:
1. Add DEEPSEEK_API_KEY to your GitHub Actions secrets and your .env file.
2. Install the openai-compatible client: `pip install openai`
3. Call `summarize_source(text)` or `synthesize_rule(evidence_list)` from
   scripts/fetch_updates.py and scripts/synthesize_best_practices.py
   respectively.
4. Replace the keyword-matching logic in synthesize_best_practices.py with
   `synthesize_rule(discovered_summaries)` for LLM-powered rule synthesis.

DEEPSEEK API NOTES:
- DeepSeek's API is compatible with the OpenAI Python client.
- Base URL: https://api.deepseek.com
- Models: deepseek-chat (general), deepseek-reasoner (chain-of-thought)
- Docs: https://platform.deepseek.com/docs
"""

from __future__ import annotations

import os


def _get_client():
    """Return an OpenAI-compatible client pointed at DeepSeek."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Run `pip install openai` to enable DeepSeek integration."
        ) from e

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Add it to GitHub Actions secrets or your local .env file."
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def summarize_source(text: str, max_words: int = 120) -> str:
    """
    Summarize a source article or abstract into a compact evidence statement.

    Args:
        text: Raw article text or abstract (up to ~4000 chars recommended).
        max_words: Target summary length.

    Returns:
        A concise summary string, or the original text if the API is unavailable.
    """
    try:
        client = _get_client()
    except (ImportError, EnvironmentError):
        # Graceful degradation: return original text truncated
        return text[:600]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst specializing in AI search and GEO (Generative Engine Optimization). "
                    "Summarize the following source into a single, dense, factual paragraph of at most "
                    f"{max_words} words. Focus on: what was measured, what was found, and what it implies for "
                    "content creators targeting AI-powered search surfaces."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def synthesize_rule(evidence_summaries: list[str]) -> dict | None:
    """
    Given a list of evidence summaries, synthesize a candidate best-practice rule.

    Args:
        evidence_summaries: List of short evidence statements from discovered sources.

    Returns:
        A dict with keys matching the best_practices.csv schema, or None if synthesis fails.
    """
    if not evidence_summaries:
        return None

    try:
        client = _get_client()
    except (ImportError, EnvironmentError):
        return None

    combined = "\n".join(f"- {s}" for s in evidence_summaries[:10])
    prompt = (
        "You are a GEO best-practice analyst. Based on the following evidence statements, "
        "generate one concise, actionable best-practice rule. "
        "Return ONLY a JSON object with these exact keys: "
        "Practice ID (use 'BP-AI-XXX' format), Rule Title, Rule Statement, Why It Matters, "
        "Confidence Level (one of: Official, Strong, Emerging, Speculative), "
        "Applies To, Does Not Apply To, Implementation Pattern.\n\n"
        f"Evidence:\n{combined}"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    import json

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None
