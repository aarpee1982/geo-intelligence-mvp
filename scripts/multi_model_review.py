"""
Multi-model GEO article reviewer.

Reviews each article section with DeepSeek, Gemini, and OpenAI sequentially.
DeepSeek-reasoner synthesizes the consensus from all three perspectives.
If a model fails, its result is skipped gracefully.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def _deepseek_client():
    from openai import OpenAI
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _openai_client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _gemini_review(section_label: str, section_text: str, rules_summary: str) -> Optional[dict]:
    """Call Gemini via REST API."""
    try:
        import urllib.request
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            return None

        prompt = _build_section_prompt(section_label, section_text, rules_summary)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[Gemini] section review failed: {e}")
        return None


def _build_section_prompt(section_label: str, section_text: str, rules_summary: str) -> str:
    return f"""You are a GEO (Generative Engine Optimization) editor. Review this article section and give specific, actionable feedback so it gets cited by AI search engines like Google AI Overviews, ChatGPT Search, and Perplexity.

GEO best practices:
{rules_summary}

Section: {section_label}
Text:
\"\"\"{section_text}\"\"\"

Return ONLY a valid JSON object with these exact keys:
- "geo_score": integer 0-100 for this section's AI-citation readiness
- "issues": list of strings, each describing one specific problem (reference the actual words)
- "suggestion": a rewritten version that fixes the issues (same meaning, better form)
- "why": one sentence explaining the most important change and why it helps AI citations"""


def review_section_deepseek(section_label: str, section_text: str, rules_summary: str) -> Optional[dict]:
    try:
        client = _deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": _build_section_prompt(section_label, section_text, rules_summary)}],
            max_tokens=800,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[DeepSeek] section review failed: {e}")
        return None


def review_section_openai(section_label: str, section_text: str, rules_summary: str) -> Optional[dict]:
    try:
        client = _openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _build_section_prompt(section_label, section_text, rules_summary)}],
            max_tokens=800,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[OpenAI] section review failed: {e}")
        return None


def review_section_gemini(section_label: str, section_text: str, rules_summary: str) -> Optional[dict]:
    return _gemini_review(section_label, section_text, rules_summary)


def synthesize_consensus(
    section_label: str,
    section_text: str,
    deepseek_result: Optional[dict],
    openai_result: Optional[dict],
    gemini_result: Optional[dict],
    rules_summary: str,
) -> dict:
    """
    DeepSeek-reasoner synthesizes consensus from available model results.
    Falls back to deepseek-chat if reasoner quota is unavailable.
    """
    available = []
    if deepseek_result:
        available.append(f"DeepSeek says:\nIssues: {deepseek_result.get('issues')}\nSuggestion: {deepseek_result.get('suggestion')}\nWhy: {deepseek_result.get('why')}")
    if openai_result:
        available.append(f"OpenAI says:\nIssues: {openai_result.get('issues')}\nSuggestion: {openai_result.get('suggestion')}\nWhy: {openai_result.get('why')}")
    if gemini_result:
        available.append(f"Gemini says:\nIssues: {gemini_result.get('issues')}\nSuggestion: {gemini_result.get('suggestion')}\nWhy: {gemini_result.get('why')}")

    if not available:
        return {"consensus_suggestion": section_text, "consensus_why": "No models returned results.", "consensus_score": None}

    if len(available) == 1:
        # Only one model succeeded — use it directly as consensus
        result = deepseek_result or openai_result or gemini_result
        return {
            "consensus_suggestion": result.get("suggestion", section_text),
            "consensus_why": result.get("why", ""),
            "consensus_score": result.get("geo_score"),
        }

    prompt = f"""You are DeepSeek-Reasoner, synthesizing GEO feedback from multiple AI models into one authoritative consensus.

Section being reviewed: {section_label}
Original text:
\"\"\"{section_text}\"\"\"

GEO best practices applied:
{rules_summary}

Individual model reviews:
{chr(10).join(available)}

Your task:
1. Find where the models AGREE — these are the highest-confidence issues
2. Resolve any DISAGREEMENTS by applying the GEO best practices as the arbiter
3. Write one definitive rewritten version of the section that satisfies all agreed improvements
4. Explain the consensus reasoning in 2 sentences

Return ONLY a valid JSON object with:
- "consensus_suggestion": the single best rewrite of the section
- "consensus_why": 2-sentence explanation of what was changed and why all models agree it improves AI citation
- "consensus_score": integer 0-100 for the rewritten section's expected GEO performance
- "agreement_points": list of the 1-3 things all models agreed on"""

    for model in ["deepseek-reasoner", "deepseek-chat"]:
        try:
            client = _deepseek_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Consensus/{model}] failed: {e}")
            continue

    # Final fallback
    return {
        "consensus_suggestion": (deepseek_result or openai_result or gemini_result or {}).get("suggestion", section_text),
        "consensus_why": "Consensus synthesis unavailable; showing best individual result.",
        "consensus_score": None,
        "agreement_points": [],
    }


def multi_model_review_section(
    section_label: str,
    section_text: str,
    rules_summary: str,
) -> dict:
    """
    Run sequential multi-model review for one section.
    Returns all individual results plus consensus.
    """
    print(f"  [DeepSeek] reviewing: {section_label}")
    ds = review_section_deepseek(section_label, section_text, rules_summary)

    print(f"  [OpenAI] reviewing: {section_label}")
    oai = review_section_openai(section_label, section_text, rules_summary)

    print(f"  [Gemini] reviewing: {section_label}")
    gem = review_section_gemini(section_label, section_text, rules_summary)

    print(f"  [Consensus] synthesizing: {section_label}")
    consensus = synthesize_consensus(section_label, section_text, ds, oai, gem, rules_summary)

    scores = [r.get("geo_score") for r in [ds, oai, gem] if r and r.get("geo_score") is not None]
    avg_score = round(sum(scores) / len(scores)) if scores else None

    return {
        "label": section_label,
        "original": section_text,
        "deepseek": ds,
        "openai": oai,
        "gemini": gem,
        "consensus": consensus,
        "avg_score": avg_score,
        "models_succeeded": sum(1 for r in [ds, oai, gem] if r is not None),
    }
