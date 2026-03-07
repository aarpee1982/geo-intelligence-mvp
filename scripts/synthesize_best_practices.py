"""
Level 2: Best-practice synthesis engine.

Reads discovered_sources.csv and uses DeepSeek (if available) to synthesize
candidate best-practice rules. Falls back to keyword matching if DeepSeek
is not configured.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
SOURCES_CSV = DATA / "sources.csv"
DISCOVERED_CSV = DATA / "discovered_sources.csv"
BEST_CSV = DATA / "best_practices.csv"

KEYWORD_RULES = {
    "citations": {
        "id": "BP-AUTO-01",
        "title": "Prioritize citation-ready sentences",
        "statement": "Express key claims as short, stand-alone factual sentences that can be lifted cleanly by AI answer systems.",
        "why": "Citation analyses repeatedly show concise, self-contained statements are easier to quote and attribute.",
    },
    "crawl": {
        "id": "BP-AUTO-02",
        "title": "Check crawler eligibility before content tweaks",
        "statement": "Verify that the platforms you want visibility from can actually crawl and surface the page.",
        "why": "Content changes do little if the relevant AI search surface cannot fetch the page.",
    },
    "retrieval": {
        "id": "BP-AUTO-03",
        "title": "Build for retrieval pathways, not just keyword rank",
        "statement": "Cover adjacent questions and retrieval paths instead of only optimizing for one exact query.",
        "why": "Modern AI answer systems fan out across related intents and source evidence from broader retrieval paths.",
    },
    "fan-out": {
        "id": "BP-AUTO-04",
        "title": "Expand content for query fan-out coverage",
        "statement": "Structure pages to answer the head query and at least 3-5 semantically adjacent sub-questions.",
        "why": "AI systems issue multiple sub-queries per user intent; pages covering more of the fan-out capture more citations.",
    },
    "structure": {
        "id": "BP-AUTO-05",
        "title": "Use explicit structural signals",
        "statement": "Use headings, FAQ blocks, and comparison tables to make content machine-parseable.",
        "why": "Structured content yields higher sentence-match rates in AI citation studies.",
    },
}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def make_rule_row(rule: dict, source: str = "AUTO keyword match") -> dict:
    return {
        "Practice ID": rule.get("Practice ID") or rule.get("id", ""),
        "Rule Title": rule.get("Rule Title") or rule.get("title", ""),
        "Rule Statement": rule.get("Rule Statement") or rule.get("statement", ""),
        "Why It Matters": rule.get("Why It Matters") or rule.get("why", ""),
        "Evidence Backing It": source,
        "Confidence Level": rule.get("Confidence Level", "Emerging"),
        "Applies To": rule.get("Applies To", "AI-search-oriented editorial content"),
        "Does Not Apply To": rule.get("Does Not Apply To", "Pure narrative or brand manifesto pages"),
        "Implementation Pattern": rule.get("Implementation Pattern", "Auto-generated; human review required."),
        "Before Example": rule.get("Before Example", ""),
        "After Example": rule.get("After Example", ""),
        "Success Metric": "Human-reviewed upgrade into canonical library",
        "Owner": "Pending review",
        "Last Reviewed": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def main() -> None:
    discovered = load_csv(DISCOVERED_CSV)
    best = load_csv(BEST_CSV)
    existing_ids = set(best.get("Practice ID", pd.Series(dtype=str)).astype(str).tolist())
    additions = []

    use_deepseek = bool(os.getenv("DEEPSEEK_API_KEY"))

    if use_deepseek and not discovered.empty:
        print("[DeepSeek] Attempting LLM-based rule synthesis...")
        try:
            from deepseek_integration import synthesize_rule
            summaries = (
                discovered.get("Summary", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .tolist()
            )
            # Batch into groups of 10 for synthesis
            for i in range(0, min(len(summaries), 30), 10):
                batch = summaries[i:i + 10]
                rule = synthesize_rule(batch)
                if rule and rule.get("Practice ID") not in existing_ids:
                    row = make_rule_row(rule, source="DeepSeek synthesis from discovered sources")
                    additions.append(row)
                    existing_ids.add(rule.get("Practice ID", ""))
            print(f"[DeepSeek] Synthesized {len(additions)} candidate rule(s).")
        except Exception as e:
            print(f"[DeepSeek] Synthesis failed, falling back to keyword match: {e}")
            use_deepseek = False

    if not use_deepseek:
        print("[Keyword] Running keyword-based rule synthesis...")
        text_blob = ""
        if not discovered.empty:
            text_blob = " ".join(
                discovered.get("Title", pd.Series(dtype=str)).astype(str).tolist()
                + discovered.get("Summary", pd.Series(dtype=str)).astype(str).tolist()
            ).lower()

        for kw, rule in KEYWORD_RULES.items():
            if kw in text_blob and rule["id"] not in existing_ids:
                additions.append(make_rule_row(rule))
                existing_ids.add(rule["id"])

    if additions:
        best = pd.concat([best, pd.DataFrame(additions)], ignore_index=True)
        save_csv(best, BEST_CSV)
        print(f"Added {len(additions)} candidate best-practice rule(s).")
    else:
        print("No new best-practice candidates created.")


if __name__ == "__main__":
    main()
