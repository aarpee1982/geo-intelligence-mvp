"""
DeepSeek API integration — wired and active.

Uses the OpenAI-compatible client pointed at DeepSeek's API.
Set DEEPSEEK_API_KEY in GitHub Actions secrets or a local .env file.

Functions:
  summarize_source(text)         -> compact evidence statement
  synthesize_rule(summaries)     -> structured best-practice dict or None
"""

from __future__ import annotations

import json
import os


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Run `pip install openai` to enable DeepSeek.") from e

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def summarize_source(text: str, max_words: int = 120) -> str:
    """
    Summarize raw article text into a dense, GEO-focused evidence statement.
    Falls back to truncated text if API is unavailable.
    """
    if not text or not text.strip():
        return ""
    try:
        client = _get_client()
    except (ImportError, EnvironmentError):
        return text[:600]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research analyst specializing in AI search and GEO "
                        "(Generative Engine Optimization). Summarize the following source "
                        f"into one dense factual paragraph of at most {max_words} words. "
                        "Focus on: what was measured or argued, what was found, and what "
                        "it implies for content creators targeting AI-powered search surfaces. "
                        "Be direct. No filler."
                    ),
                },
                {"role": "user", "content": text[:4000]},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DeepSeek] summarize_source failed: {e}")
        return text[:600]


def synthesize_rule(evidence_summaries: list[str]) -> dict | None:
    """
    Given a list of evidence summaries, synthesize one candidate best-practice rule.
    Returns a dict matching best_practices.csv schema, or None on failure.
    """
    if not evidence_summaries:
        return None

    try:
        client = _get_client()
    except (ImportError, EnvironmentError):
        return None

    combined = "\n".join(f"- {s}" for s in evidence_summaries[:10])
    prompt = (
        "You are a GEO best-practice analyst. Based on the evidence statements below, "
        "generate one concise, actionable best-practice rule for content teams targeting "
        "AI-powered search surfaces (Google AI Overviews, ChatGPT Search, Perplexity). "
        "Return ONLY a valid JSON object with these exact keys:\n"
        "  Practice ID (format: BP-AI-XXX, pick a unique 3-digit number),\n"
        "  Rule Title (max 8 words),\n"
        "  Rule Statement (one clear actionable sentence),\n"
        "  Why It Matters (one sentence grounded in the evidence),\n"
        "  Confidence Level (one of: Official, Strong, Emerging, Speculative),\n"
        "  Applies To (content types this applies to),\n"
        "  Does Not Apply To (exceptions),\n"
        "  Implementation Pattern (one concrete how-to sentence).\n\n"
        f"Evidence:\n{combined}"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[DeepSeek] synthesize_rule failed: {e}")
        return None
