from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES_CSV = DATA / "sources.csv"
DISCOVERED_CSV = DATA / "discovered_sources.csv"
BEST_CSV = DATA / "best_practices.csv"

KEYWORD_RULES = {
    "citations": {
        "id": "BP-AUTO-01",
        "title": "Prioritize citation-ready sentences",
        "statement": "Express key claims as short, stand-alone factual sentences that can be lifted cleanly by AI answer systems.",
        "why": "Recent citation analyses repeatedly show concise, self-contained statements are easier to quote and attribute.",
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
}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    sources = load_csv(SOURCES_CSV)
    discovered = load_csv(DISCOVERED_CSV)
    best = load_csv(BEST_CSV)

    text_blob = " ".join(discovered.get("Title", pd.Series(dtype=str)).astype(str).tolist() + discovered.get("Summary", pd.Series(dtype=str)).astype(str).tolist()).lower()
    additions = []
    for kw, rule in KEYWORD_RULES.items():
        if kw in text_blob and (best.get("Practice ID", pd.Series(dtype=str)) != rule["id"]).all():
            additions.append({
                "Practice ID": rule["id"],
                "Rule Title": rule["title"],
                "Rule Statement": rule["statement"],
                "Why It Matters": rule["why"],
                "Evidence Backing It": "AUTO from discovered sources",
                "Confidence Level": "Emerging",
                "Applies To": "AI-search-oriented editorial content",
                "Does Not Apply To": "Pure narrative or brand manifesto pages",
                "Implementation Pattern": "Auto-generated from monitored source signals; human review required.",
                "Before Example": "Long, blended claims with weak evidence cues.",
                "After Example": "Short source-backed statements and stronger retrieval coverage.",
                "Success Metric": "Human-reviewed upgrade into canonical library",
                "Owner": "Pending review",
                "Last Reviewed": datetime.utcnow().strftime("%Y-%m-%d"),
            })
    if additions:
        best = pd.concat([best, pd.DataFrame(additions)], ignore_index=True)
        save_csv(best, BEST_CSV)
        print(f"Added {len(additions)} candidate best-practice rules.")
    else:
        print("No new best-practice candidates created.")


if __name__ == "__main__":
    main()
