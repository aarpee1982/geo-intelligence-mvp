# Deployment Checklist

## 1. What to Upload

Push the full repository to GitHub. The following files and folders must be present:

```
app/app.py
app/utils.py
data/sources.csv
data/best_practices.csv
data/review_template.csv
config/sources.yaml
scripts/fetch_updates.py
scripts/synthesize_best_practices.py
scripts/rebuild_excel.py
scripts/deepseek_integration.py
.github/workflows/daily_update.yml
.streamlit/config.toml
streamlit_app.py
requirements.txt
.gitignore
README.md
DEPLOYMENT_CHECKLIST.md
```

Do NOT commit:
- `.env` files
- `__pycache__/` directories
- `data/feed_cache/` (created at runtime)
- `.streamlit/secrets.toml` (if you create one locally)

---

## 2. Secrets to Add

For the base pipeline (RSS feeds only), no secrets are needed.

When you activate DeepSeek:

| Secret Name | Where to Get It | Where to Add It |
|---|---|---|
| `DEEPSEEK_API_KEY` | platform.deepseek.com | GitHub repo > Settings > Secrets and variables > Actions |

For local development, create a `.env` file at the repo root:
```
DEEPSEEK_API_KEY=your-key-here
```

---

## 3. How to Enable GitHub Actions

1. Push the repo to GitHub
2. Go to your repo > **Actions** tab
3. If prompted, click **"I understand my workflows, go ahead and enable them"**
4. The `Daily GEO Update` workflow will now run automatically every day at 03:15 UTC
5. To trigger it manually: Actions > Daily GEO Update > **Run workflow**

---

## 4. How to Run the Streamlit App Locally

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

App opens at http://localhost:8501

To run the data pipeline scripts independently:
```bash
python scripts/fetch_updates.py
python scripts/synthesize_best_practices.py
python scripts/rebuild_excel.py
```

---

## 5. Next Development Priorities

In order of leverage:

**P1: Activate DeepSeek summarization**
- Enables LLM-quality rule synthesis instead of keyword matching
- Wire `deepseek_integration.py` into `fetch_updates.py` and `synthesize_best_practices.py`
- Estimated effort: 2-3 hours

**P2: Source approval workflow in the UI**
- Currently, promoting a candidate from `discovered_sources.csv` to `sources.csv` requires a manual CSV edit
- Add a simple "Approve" button in the Source Library tab that moves rows between files
- Estimated effort: half a day

**P3: Expand the RSS watchlist**
- Add more feeds to `config/sources.yaml`: Perplexity blog, Bing Webmaster Blog, Anthropic news, AI research newsletters
- No code change needed; just edit the YAML

**P4: Article reviewer depth**
- Current scoring is heuristic (keyword-based)
- Upgrade: send the draft to DeepSeek with the best-practice rules as context; return structured feedback
- Estimated effort: 1 day once DeepSeek is wired in

**P5: Streamlit Cloud deployment**
- The app is Streamlit Cloud-ready as-is
- Go to share.streamlit.io, connect the GitHub repo, set the entrypoint to `streamlit_app.py`
- Add `DEEPSEEK_API_KEY` to Streamlit Cloud secrets when ready
