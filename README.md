# GEO Intelligence Hub

An internal MVP for **Generative Engine Optimization (GEO)** intelligence. It ingests authoritative sources on AI search, AI citations, and query fan-out; synthesizes canonical best practices; and reviews article drafts against those practices.

---

## Architecture

The system is a three-level pipeline:

```
Level 1 — Source Ingestion
  RSS watchlist (config/sources.yaml)  +  boss/team manual intake (app UI)
        |
        v
  data/discovered_sources.csv  -->  data/sources.csv  (approved)
        |
Level 2 — Best Practice Engine
  scripts/synthesize_best_practices.py
        |
        v
  data/best_practices.csv
        |
Level 3 — Article Reviewer
  Streamlit UI (Article Reviewer tab)
        |
        v
  data/reviews_log.csv  +  exports/GEO_Intelligence_Live.xlsx
```

**Storage:** flat CSV files (no database, no spreadsheet-as-primary-DB).  
**Interface:** Streamlit multi-tab app.  
**Automation:** GitHub Actions runs the pipeline daily and commits outputs.  
**Export:** Excel workbook is a generated artifact, rebuilt on each run.

---

## Folder Structure

```
geo-intelligence-mvp/
├── app/
│   ├── app.py              # Main Streamlit app (4 tabs)
│   └── utils.py            # Shared helpers: CSV I/O, ID generation, workbook builder
├── data/
│   ├── sources.csv         # Approved, seeded evidence library (Level 1)
│   ├── best_practices.csv  # Canonical best-practice rules (Level 2)
│   ├── review_template.csv # Column schema for review logs
│   ├── discovered_sources.csv  # Auto-discovered candidates (created at runtime)
│   ├── manual_submissions.csv  # Boss/team intake queue (created at runtime)
│   └── reviews_log.csv     # Article review history (created at runtime)
├── exports/
│   └── GEO_Intelligence_Live.xlsx  # Auto-rebuilt Excel export
├── scripts/
│   ├── fetch_updates.py             # Fetches RSS feeds; writes discovered_sources.csv
│   ├── synthesize_best_practices.py # Keyword-based rule synthesis (DeepSeek-upgradeable)
│   ├── rebuild_excel.py             # Writes all CSVs into the Excel workbook
│   └── deepseek_integration.py     # PLACEHOLDER: DeepSeek API wiring (not yet active)
├── config/
│   └── sources.yaml        # RSS watchlist with trust scores
├── .github/workflows/
│   └── daily_update.yml    # GitHub Actions: fetch, synthesize, export, commit
├── .streamlit/
│   └── config.toml         # Streamlit server and theme config
├── streamlit_app.py        # Root-level entrypoint for streamlit run
├── requirements.txt
├── .gitignore
├── README.md
└── DEPLOYMENT_CHECKLIST.md
```

---

## Running Locally

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/geo-intelligence-mvp.git
cd geo-intelligence-mvp

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501

**To run the pipeline scripts manually:**

```bash
python scripts/fetch_updates.py
python scripts/synthesize_best_practices.py
python scripts/rebuild_excel.py
```

---

## App Tabs

| Tab | Purpose |
|---|---|
| **Add Source** | Boss-friendly form to submit new URLs with context and priority |
| **Source Library** | Searchable view of the approved evidence base; download Excel |
| **Best Practices** | Filterable canonical rules with confidence levels |
| **Article Reviewer** | Paste a draft; get a GEO score, violations, and specific edits |

---

## GitHub Actions

The workflow at `.github/workflows/daily_update.yml`:
- Triggers daily at 03:15 UTC and on manual workflow_dispatch
- Runs the full pipeline: fetch RSS feeds, synthesize rules, rebuild Excel
- Commits any changed CSVs and the Excel file back to the repo automatically

To enable: push to GitHub, go to the Actions tab, and enable workflows if prompted.

---

## Adding GitHub Secrets

Currently no secrets are required for the base pipeline (all RSS feeds are public).

When DeepSeek is wired in, add this secret:

```
Settings > Secrets and variables > Actions > New repository secret
Name:  DEEPSEEK_API_KEY
Value: your-deepseek-api-key
```

For local use, create a `.env` file (already in .gitignore):

```
DEEPSEEK_API_KEY=your-deepseek-api-key
```

---

## Plugging in DeepSeek

`scripts/deepseek_integration.py` contains two ready-to-use functions:

- `summarize_source(text)` condenses a crawled article into a dense evidence statement
- `synthesize_rule(evidence_summaries)` generates a structured best-practice rule from evidence

To activate:
1. Add `openai` to requirements.txt
2. Set `DEEPSEEK_API_KEY` as described above
3. In `fetch_updates.py`, call `summarize_source` on each fetched article body
4. In `synthesize_best_practices.py`, replace the keyword-match block with `synthesize_rule`

Both functions degrade gracefully when the key is absent, so you can wire them in incrementally.

---

## Boss Intake Flow

1. Boss or team member opens the **Add Source** tab
2. Pastes a URL, adds a short note on why it matters, sets priority
3. Submission is stored in `data/manual_submissions.csv`
4. Next pipeline run picks it up; manual promotion to `sources.csv` completes it

No login required. No external service. All data stays in the repo.

---

## What Is Placeholder vs Functional

| Component | Status |
|---|---|
| Source ingestion from RSS | Functional |
| Manual intake form | Functional |
| Best-practice library (seeded) | Functional (10 seeded rules) |
| Keyword-based rule synthesis | Functional (basic heuristic) |
| Article reviewer + scoring | Functional (heuristic scoring) |
| Excel export | Functional |
| GitHub Actions pipeline | Functional |
| DeepSeek summarization | Placeholder (wiring instructions in deepseek_integration.py) |
| Source approval workflow | Partial (manual CSV edit to promote candidates) |

---

## License

Internal MVP. Not for public distribution.
