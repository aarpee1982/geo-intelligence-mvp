from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORTS_DIR = ROOT / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

SOURCES_CSV = DATA_DIR / "sources.csv"
BEST_PRACTICES_CSV = DATA_DIR / "best_practices.csv"
REVIEWS_CSV = DATA_DIR / "reviews_log.csv"
MANUAL_SUBMISSIONS_CSV = DATA_DIR / "manual_submissions.csv"
REVIEW_TEMPLATE_CSV = DATA_DIR / "review_template.csv"


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def url_hash(url: str) -> str:
    return hashlib.md5(url.strip().lower().encode("utf-8")).hexdigest()[:10]


def next_id(prefix: str, series: Iterable[str]) -> str:
    nums = []
    for value in series:
        if isinstance(value, str) and value.startswith(prefix):
            try:
                nums.append(int(value.split("-")[-1]))
            except Exception:
                pass
    nxt = max(nums, default=0) + 1
    return f"{prefix}-{nxt:03d}"


def ensure_manual_submission_table() -> pd.DataFrame:
    cols = [
        "Submission ID",
        "Submitted At",
        "Submitted By",
        "URL",
        "Title / Note",
        "Why It Matters",
        "Priority",
        "Status",
        "Dedup Key",
    ]
    df = load_csv(MANUAL_SUBMISSIONS_CSV)
    if df.empty:
        df = pd.DataFrame(columns=cols)
        save_csv(df, MANUAL_SUBMISSIONS_CSV)
    return df


def ensure_reviews_table() -> pd.DataFrame:
    df = load_csv(REVIEWS_CSV)
    if df.empty:
        template = load_csv(REVIEW_TEMPLATE_CSV)
        df = template.iloc[0:0].copy()
        save_csv(df, REVIEWS_CSV)
    return df


def workbook_bytes() -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)

    sheets = [
        ("Level1_Sources", load_csv(SOURCES_CSV)),
        ("Level2_BestPractices", load_csv(BEST_PRACTICES_CSV)),
        ("Level3_Reviews", load_csv(REVIEWS_CSV)),
        ("Manual_Submissions", load_csv(MANUAL_SUBMISSIONS_CSV)),
    ]
    for name, df in sheets:
        ws = wb.create_sheet(name)
        if df.empty:
            ws.append(["No data"])
            continue
        ws.append(list(df.columns))
        for row in df.fillna("").itertuples(index=False):
            ws.append(list(row))

    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.read()
