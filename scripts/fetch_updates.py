"""
Level 1: RSS feed ingestion with optional DeepSeek summarization.

Reads config/sources.yaml, fetches each RSS feed, scores entries,
and writes new candidates to data/discovered_sources.csv.
If DEEPSEEK_API_KEY is set, uses DeepSeek to summarize each entry.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data"
CONFIG = ROOT / "config" / "sources.yaml"
RAW_DIR = DATA / "feed_cache"
RAW_DIR.mkdir(exist_ok=True)

SOURCES_CSV = DATA / "sources.csv"
DISCOVERED_CSV = DATA / "discovered_sources.csv"

USE_DEEPSEEK = bool(os.getenv("DEEPSEEK_API_KEY"))


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def score_item(source_cfg: dict, entry: dict) -> float:
    title = (entry.get("title") or "").lower()
    summary = (entry.get("summary") or "").lower()
    text = title + " " + summary
    score = float(source_cfg.get("trust_score", 0.5)) * 50
    for kw in ["ai overview", "ai mode", "chatgpt", "llm", "crawl", "citation", "retrieval", "search", "geo", "generative"]:
        if kw in text:
            score += 8
    if any(kw in text for kw in ["study", "research", "analysis", "paper", "data"]):
        score += 10
    return min(score, 100)


def get_summary(source_cfg: dict, entry: dict) -> str:
    raw = (entry.get("summary") or "")[:800]
    if not USE_DEEPSEEK or not raw.strip():
        return raw
    try:
        from deepseek_integration import summarize_source
        return summarize_source(raw)
    except Exception as e:
        print(f"[DeepSeek] summarize failed for '{entry.get('title', '')}': {e}")
        return raw


def normalize_entry(source_cfg: dict, entry: dict) -> dict:
    try:
        dedup_date = (
            datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d")
            if getattr(entry, "published_parsed", None)
            else ""
        )
    except Exception:
        dedup_date = ""

    score = score_item(source_cfg, entry)
    summary = get_summary(source_cfg, entry)

    return {
        "Discovered At": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Feed Name": source_cfg["name"],
        "Feed Category": source_cfg["category"],
        "Title": entry.get("title", ""),
        "URL": entry.get("link", ""),
        "Published Date": dedup_date,
        "Summary": summary,
        "Platform Relevance": source_cfg.get("platform_relevance", ""),
        "Authority Score": round(score, 1),
        "Status": "candidate",
    }


def main() -> None:
    cfg = load_yaml(CONFIG)
    discovered = load_csv(DISCOVERED_CSV)
    existing_urls = set(load_csv(SOURCES_CSV).get("URL", pd.Series(dtype=str)).astype(str).tolist())
    known_urls = set(discovered.get("URL", pd.Series(dtype=str)).astype(str).tolist()) | existing_urls

    if USE_DEEPSEEK:
        print("[DeepSeek] Summarization enabled.")
    else:
        print("[Fetch] Running without DeepSeek (DEEPSEEK_API_KEY not set).")

    rows = []
    for source_cfg in cfg.get("watchlist", []):
        print(f"  Fetching: {source_cfg['name']}")
        try:
            feed = feedparser.parse(source_cfg["url"])
            cache_path = RAW_DIR / (source_cfg["name"].replace(" ", "_").lower() + ".json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(feed.get("feed", {}), f, ensure_ascii=False, indent=2)
            for entry in feed.entries[:20]:
                row = normalize_entry(source_cfg, entry)
                url = row["URL"]
                if url and url not in known_urls:
                    rows.append(row)
                    known_urls.add(url)
        except Exception as e:
            print(f"  [Error] {source_cfg['name']}: {e}")

    if rows:
        discovered = pd.concat([discovered, pd.DataFrame(rows)], ignore_index=True)
        discovered = discovered.sort_values(
            by=["Authority Score", "Discovered At"], ascending=[False, False]
        )
        save_csv(discovered, DISCOVERED_CSV)
        print(f"\nAdded {len(rows)} new candidate source(s).")
    else:
        print("\nNo new candidate sources found.")


if __name__ == "__main__":
    main()
