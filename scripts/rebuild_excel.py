from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPORTS = ROOT / "exports"
EXPORTS.mkdir(exist_ok=True)
OUT = EXPORTS / "GEO_Intelligence_Live.xlsx"

SHEETS = {
    "Level1_Sources": DATA / "sources.csv",
    "Level1_Candidates": DATA / "discovered_sources.csv",
    "Level2_BestPractices": DATA / "best_practices.csv",
    "Level3_Reviews": DATA / "reviews_log.csv",
    "Manual_Submissions": DATA / "manual_submissions.csv",
}


def read_df(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    wb = Workbook()
    wb.remove(wb.active)
    for sheet_name, csv_path in SHEETS.items():
        df = read_df(csv_path)
        ws = wb.create_sheet(sheet_name)
        if df.empty:
            ws.append(["No data"])
            continue
        ws.append(list(df.columns))
        for row in df.fillna("").itertuples(index=False):
            ws.append(list(row))
    wb.save(OUT)
    print(f"Workbook rebuilt at {OUT}")


if __name__ == "__main__":
    main()
