from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make scripts/ importable from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils import (
    BEST_PRACTICES_CSV,
    DISCOVERED_CSV,
    MANUAL_SUBMISSIONS_CSV,
    REVIEWS_CSV,
    SOURCES_CSV,
    ensure_manual_submission_table,
    ensure_reviews_table,
    load_csv,
    next_id,
    now_iso,
    save_csv,
    url_hash,
    workbook_bytes,
)

st.set_page_config(page_title="GEO Intelligence Hub", layout="wide")
st.title("GEO Intelligence Hub")
st.caption("Source ingestion, best-practice library, and article reviewer for AI search.")

# ── Model availability ────────────────────────────────────────────────────────

HAS_DEEPSEEK = bool(os.getenv("DEEPSEEK_API_KEY"))
HAS_OPENAI   = bool(os.getenv("OPENAI_API_KEY"))
HAS_GEMINI   = bool(os.getenv("GEMINI_API_KEY"))
MULTI_MODE   = HAS_DEEPSEEK and (HAS_OPENAI or HAS_GEMINI)

# ── Auto-approve helpers ──────────────────────────────────────────────────────

AUTO_APPROVE_SCORE = 60
AUTO_APPROVE_CATEGORIES = {"official", "research", "vendor_research", "practitioner", "analyst"}


def promote_to_library(row: pd.Series, approved_df: pd.DataFrame, source: str = "auto") -> pd.DataFrame:
    existing_urls = set(approved_df.get("URL", pd.Series(dtype=str)).astype(str).tolist())
    if row.get("URL", "") in existing_urls:
        return approved_df
    new_id = next_id("SRC", approved_df.get("Source ID", pd.Series(dtype=str)))
    new_row = {
        "Source ID": new_id,
        "Title": row.get("Title", ""),
        "URL": row.get("URL", ""),
        "Source Type": "Auto-approved / pending human review" if source == "auto" else "Boss submission",
        "Publisher": row.get("Feed Name", ""),
        "Author": "",
        "Publication Date": row.get("Published Date", ""),
        "Platform Relevance": row.get("Platform Relevance", ""),
        "Evidence Type": "Auto-discovered",
        "Sample Size / Scope": "",
        "Core Claim": str(row.get("Summary", ""))[:300],
        "Extracted Findings": row.get("Summary", ""),
        "Operational Implications": "",
        "Confidence Level": "Emerging",
        "Contradictions / Caveats": "",
        "Tags": row.get("Feed Category", ""),
        "Last Reviewed": now_iso()[:10],
    }
    return pd.concat([approved_df, pd.DataFrame([new_row])], ignore_index=True)


def bulk_auto_approve(discovered: pd.DataFrame):
    approved_df = load_csv(SOURCES_CSV)
    candidates = discovered[discovered.get("Status", pd.Series(dtype=str)) == "candidate"].copy() if "Status" in discovered.columns else discovered.copy()
    eligible = candidates[
        (candidates["Authority Score"].astype(float) >= AUTO_APPROVE_SCORE) &
        (candidates["Feed Category"].isin(AUTO_APPROVE_CATEGORIES))
    ]
    count = 0
    for _, row in eligible.iterrows():
        approved_df = promote_to_library(row, approved_df, source="auto")
        discovered.loc[discovered["URL"] == row["URL"], "Status"] = "approved"
        count += 1
    return approved_df, discovered, count


def auto_approve_boss_submissions(submissions: pd.DataFrame):
    approved_df = load_csv(SOURCES_CSV)
    new_subs = submissions[submissions.get("Status", pd.Series(dtype=str)) == "new"]
    count = 0
    for _, row in new_subs.iterrows():
        fake_row = pd.Series({
            "Title": row.get("Title / Note", ""),
            "URL": row.get("URL", ""),
            "Feed Name": f"Boss submission by {row.get('Submitted By', 'Unknown')}",
            "Feed Category": "boss_submission",
            "Published Date": row.get("Submitted At", "")[:10],
            "Platform Relevance": "",
            "Summary": row.get("Why It Matters", ""),
            "Authority Score": 100,
        })
        approved_df = promote_to_library(fake_row, approved_df, source="boss")
        submissions.loc[submissions["Submission ID"] == row["Submission ID"], "Status"] = "approved"
        count += 1
    return approved_df, submissions, count


# ── Section splitter ──────────────────────────────────────────────────────────

def split_into_sections(text: str) -> list[dict]:
    lines = text.strip().split("\n")
    sections, current_label, current_lines, para_count = [], "Opening", [], 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                sections.append({"label": current_label, "text": " ".join(current_lines).strip()})
                current_lines = []
                para_count += 1
                current_label = f"Paragraph {para_count}"
        elif stripped.startswith("#"):
            if current_lines:
                sections.append({"label": current_label, "text": " ".join(current_lines).strip()})
                current_lines = []
            current_label = f"Heading: {stripped.lstrip('#').strip()}"
        else:
            current_lines.append(stripped)
    if current_lines:
        sections.append({"label": current_label, "text": " ".join(current_lines).strip()})
    return [s for s in sections if len(s["text"].split()) >= 8]


def get_rules_summary(practices: pd.DataFrame) -> str:
    if practices.empty:
        return ""
    return "\n".join(
        f"- {r['Rule Title']}: {r['Rule Statement']}"
        for _, r in practices.head(10).iterrows()
        if pd.notna(r.get("Rule Title")) and pd.notna(r.get("Rule Statement"))
    )


# ── Heuristic fallback ────────────────────────────────────────────────────────

def score_draft_heuristic(text: str, practices: pd.DataFrame) -> dict:
    text_low = text.lower()
    words = re.findall(r"\w+", text)
    first_120 = " ".join(words[:120]).lower()
    matched, violated, suggestions = [], [], []

    if words:
        if len(first_120.split()) > 20 and not any(k in first_120 for k in ["in short", "bottom line", "key takeaway", "the answer", "summary"]):
            violated.append("BP-01"); suggestions.append("Add a direct answer in the opening 2-4 lines.")
        else:
            matched.append("BP-01")

    if [s for s in re.split(r"[.!?]\s+", text) if len(s.split()) > 28]:
        violated.append("BP-02"); suggestions.append("Split long sentences into shorter factual claims.")
    else:
        matched.append("BP-02")

    if not any(m in text_low for m in ["##", "###", "faq", "table", "key takeaways"]):
        violated.append("BP-03"); suggestions.append("Add subheads, FAQ blocks, or tables.")
    else:
        matched.append("BP-03")

    if not any(m in text_low for m in ["source", "study", "data", "according to", "official"]):
        violated.append("BP-06"); suggestions.append("Anchor claims in explicit evidence.")
    else:
        matched.append("BP-06")

    score = max(35, min(100, 100 - 8 * len(set(violated))))
    rationale = []
    for bp in sorted(set(violated)):
        row = practices[practices["Practice ID"] == bp]
        if not row.empty:
            r = row.iloc[0]
            rationale.append(f"{bp} — {r['Rule Title']}: {r['Why It Matters']}")

    return {"matched": sorted(set(matched)), "violated": sorted(set(violated)),
            "suggestions": suggestions, "score": score, "rationale": rationale,
            "sections": [], "executive_summary": "", "mode": "heuristic"}


# ── Executive summary ─────────────────────────────────────────────────────────

def deepseek_executive_summary(section_results: list, rules_summary: str) -> str:
    try:
        from openai import OpenAI
        import json
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return ""
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        scores = [r.get("avg_score") for r in section_results if r.get("avg_score") is not None]
        avg = round(sum(scores) / len(scores)) if scores else "N/A"
        issues = []
        for r in section_results:
            for model_key in ["deepseek", "openai", "gemini"]:
                for issue in (r.get(model_key) or {}).get("issues", []):
                    issues.append(issue)
        prompt = f"""You are a senior GEO strategist. Write a 3-5 sentence executive summary of this article's AI-citation readiness.

Average section GEO score: {avg}/100
Sections reviewed: {len(section_results)}
Key issues (from all models): {json.dumps(issues[:12])}
GEO best practices: {rules_summary}

Be direct. Start with the overall verdict. Name the 2-3 most important fixes. End with one sentence on what the article achieves after those fixes. Plain paragraphs, no bullet points."""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Executive summary unavailable: {e}"


# ── Main review orchestrator ──────────────────────────────────────────────────

def run_review(text: str, practices: pd.DataFrame) -> dict:
    if not HAS_DEEPSEEK and not HAS_OPENAI and not HAS_GEMINI:
        return {**score_draft_heuristic(text, practices), "mode": "heuristic"}

    rules_summary = get_rules_summary(practices)
    sections = split_into_sections(text)
    section_results = []

    try:
        from multi_model_review import multi_model_review_section
        use_multi = MULTI_MODE
    except ImportError:
        use_multi = False

    for sec in sections:
        if use_multi:
            result = multi_model_review_section(sec["label"], sec["text"], rules_summary)
        else:
            # Single model fallback
            from multi_model_review import review_section_deepseek, review_section_openai, review_section_gemini
            ds = review_section_deepseek(sec["label"], sec["text"], rules_summary) if HAS_DEEPSEEK else None
            oai = review_section_openai(sec["label"], sec["text"], rules_summary) if HAS_OPENAI else None
            gem = review_section_gemini(sec["label"], sec["text"], rules_summary) if HAS_GEMINI else None
            scores = [r.get("geo_score") for r in [ds, oai, gem] if r and r.get("geo_score") is not None]
            result = {
                "label": sec["label"], "original": sec["text"],
                "deepseek": ds, "openai": oai, "gemini": gem,
                "consensus": ds or oai or gem or {},
                "avg_score": round(sum(scores)/len(scores)) if scores else None,
                "models_succeeded": sum(1 for r in [ds, oai, gem] if r),
            }
        section_results.append(result)

    exec_summary = deepseek_executive_summary(section_results, rules_summary)
    scores = [r["avg_score"] for r in section_results if r.get("avg_score") is not None]
    overall = round(sum(scores) / len(scores)) if scores else 50

    return {
        "score": overall, "sections": section_results,
        "executive_summary": exec_summary,
        "matched": [], "violated": [], "suggestions": [], "rationale": [],
        "mode": "multi_model" if use_multi else "single_model",
    }


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_submission(submitter, url, title, why, priority):
    df = ensure_manual_submission_table()
    dedup = url_hash(url)
    if not df.empty and dedup in df.get("Dedup Key", pd.Series(dtype=str)).astype(str).tolist():
        st.warning("This link already exists in the intake queue.")
        return
    submission_id = next_id("SUB", df.get("Submission ID", pd.Series(dtype=str)))
    row = {"Submission ID": submission_id, "Submitted At": now_iso(), "Submitted By": submitter,
           "URL": url, "Title / Note": title, "Why It Matters": why, "Priority": priority,
           "Status": "new", "Dedup Key": dedup}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, MANUAL_SUBMISSIONS_CSV)
    approved_df, _, _ = auto_approve_boss_submissions(df)
    save_csv(approved_df, SOURCES_CSV)
    save_csv(df, MANUAL_SUBMISSIONS_CSV)
    st.success(f"Saved {submission_id} and auto-approved into the Source Library.")


def save_review(title, topic, content_type, primary_platform, result):
    df = ensure_reviews_table()
    review_id = next_id("REV", df.get("Review ID", pd.Series(dtype=str)))
    row = {"Review ID": review_id, "Draft Title": title, "Target Query / Topic": topic,
           "Content Type": content_type, "Primary Platform": primary_platform,
           "Matched Best Practices": "; ".join(result.get("matched", [])),
           "Violated Best Practices": "; ".join(result.get("violated", [])),
           "Suggested Edits": " | ".join(result.get("suggestions", [])),
           "Applied Edits": "", "What Changed": "",
           "Why It Changed": " | ".join(result.get("rationale", [])),
           "Evidence Links Used": "", "Human Override Notes": "",
           "Final Score / 100": result.get("score", ""),
           "Reviewed Date": now_iso()[:10]}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, REVIEWS_CSV)


# ── Load data ─────────────────────────────────────────────────────────────────

sources    = load_csv(SOURCES_CSV)
practices  = load_csv(BEST_PRACTICES_CSV)
submissions = ensure_manual_submission_table()
reviews    = ensure_reviews_table()
discovered = load_csv(DISCOVERED_CSV)

# ── KPI bar ───────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Approved sources", len(load_csv(SOURCES_CSV)))
k2.metric("Best practices", len(practices))
k3.metric("Manual submissions", len(submissions))
k4.metric("Reviews logged", len(reviews))
pending = len(discovered[discovered["Status"] == "candidate"]) if not discovered.empty and "Status" in discovered.columns else 0
k5.metric("Candidates pending", pending)

# ── Tabs ──────────────────────────────────────────────────────────────────────

intake_tab, library_tab, candidates_tab, practice_tab, review_tab = st.tabs([
    "Add Source", "Source Library", "Approve Candidates", "Best Practices", "Article Reviewer",
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────

with intake_tab:
    st.subheader("Boss-friendly source intake")
    st.info("Any source submitted here is automatically approved into the Source Library immediately.")
    with st.form("source_intake"):
        submitter = st.text_input("Submitted by", placeholder="Rahul or Boss")
        url       = st.text_input("Source URL", placeholder="Paste the article, study, or PDF link")
        title     = st.text_input("Title or note", placeholder="What is this about?")
        why       = st.text_area("Why it matters", placeholder="One or two lines on why this is important for GEO")
        priority  = st.selectbox("Priority", ["low", "medium", "high"])
        if st.form_submit_button("Save and approve source"):
            if not url.strip():
                st.error("URL is required.")
            else:
                save_submission(submitter.strip() or "Unknown", url.strip(), title.strip(), why.strip(), priority)

# ── Tab 2 ─────────────────────────────────────────────────────────────────────

with library_tab:
    st.subheader("Approved evidence library")
    search = st.text_input("Filter by title, publisher, tag, or platform")
    view = load_csv(SOURCES_CSV)
    if search.strip():
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False))
        view = view[mask.any(axis=1)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download live workbook", data=workbook_bytes(),
                       file_name="geo_intelligence_live.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────

with candidates_tab:
    st.subheader("Candidate sources pending review")
    if discovered.empty:
        st.info("No candidate sources yet. The daily pipeline will populate this after the first run.")
    else:
        candidates = discovered[discovered["Status"] == "candidate"].copy() if "Status" in discovered.columns else discovered.copy()
        if candidates.empty:
            st.success("All discovered sources have been reviewed.")
        else:
            eligible_count = len(candidates[
                (candidates["Authority Score"].astype(float) >= AUTO_APPROVE_SCORE) &
                (candidates["Feed Category"].isin(AUTO_APPROVE_CATEGORIES))
            ])
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{eligible_count} candidates** eligible for bulk auto-approve (score 60+).")
            with col2:
                if st.button(f"Bulk auto-approve {eligible_count} candidates", type="primary"):
                    approved_df, updated_discovered, count = bulk_auto_approve(discovered)
                    save_csv(approved_df, SOURCES_CSV)
                    save_csv(updated_discovered, DISCOVERED_CSV)
                    st.success(f"Auto-approved {count} sources.")
                    st.rerun()
            st.divider()
            filter_feed = st.multiselect("Filter by feed", sorted(candidates["Feed Name"].dropna().unique().tolist()))
            if filter_feed:
                candidates = candidates[candidates["Feed Name"].isin(filter_feed)]
            st.dataframe(candidates, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("**Manually approve a single candidate**")
            candidate_urls = candidates["URL"].dropna().tolist()
            if candidate_urls:
                selected_url = st.selectbox("Select URL to approve", candidate_urls)
                selected_row = candidates[candidates["URL"] == selected_url].iloc[0]
                with st.form("approve_form"):
                    st.write(f"**{selected_row.get('Title', '')}**")
                    st.write(str(selected_row.get("Summary", ""))[:300])
                    source_type = st.selectbox("Source type", ["Study / Article", "Vendor study / Article", "Official documentation", "Research paper", "Industry report"])
                    confidence  = st.selectbox("Confidence level", ["Strong", "Official", "Emerging", "Speculative"])
                    core_claim  = st.text_area("Core claim", value=str(selected_row.get("Summary", ""))[:200])
                    tags        = st.text_input("Tags (comma separated)")
                    if st.form_submit_button("Approve and add to Source Library"):
                        approved_df = load_csv(SOURCES_CSV)
                        new_id = next_id("SRC", approved_df.get("Source ID", pd.Series(dtype=str)))
                        new_row = {
                            "Source ID": new_id, "Title": selected_row.get("Title", ""),
                            "URL": selected_url, "Source Type": source_type,
                            "Publisher": selected_row.get("Feed Name", ""), "Author": "",
                            "Publication Date": selected_row.get("Published Date", ""),
                            "Platform Relevance": selected_row.get("Platform Relevance", ""),
                            "Evidence Type": "Observational / auto-discovered", "Sample Size / Scope": "",
                            "Core Claim": core_claim, "Extracted Findings": selected_row.get("Summary", ""),
                            "Operational Implications": "", "Confidence Level": confidence,
                            "Contradictions / Caveats": "", "Tags": tags, "Last Reviewed": now_iso()[:10],
                        }
                        approved_df = pd.concat([approved_df, pd.DataFrame([new_row])], ignore_index=True)
                        save_csv(approved_df, SOURCES_CSV)
                        discovered.loc[discovered["URL"] == selected_url, "Status"] = "approved"
                        save_csv(discovered, DISCOVERED_CSV)
                        st.success(f"Approved as {new_id}.")
                        st.rerun()

# ── Tab 4 ─────────────────────────────────────────────────────────────────────

with practice_tab:
    st.subheader("Canonical best-practice library")
    confidence_opts = sorted(practices["Confidence Level"].dropna().unique().tolist()) if not practices.empty else []
    confidence = st.multiselect("Filter by confidence", confidence_opts, default=confidence_opts)
    pview = practices[practices["Confidence Level"].isin(confidence)] if confidence else practices
    st.dataframe(pview, use_container_width=True, hide_index=True)

# ── Tab 5 ─────────────────────────────────────────────────────────────────────

with review_tab:
    st.subheader("Article reviewer")

    # Model status bar
    model_cols = st.columns(4)
    model_cols[0].markdown(f"{'🟢' if HAS_DEEPSEEK else '🔴'} **DeepSeek**")
    model_cols[1].markdown(f"{'🟢' if HAS_OPENAI else '🔴'} **OpenAI**")
    model_cols[2].markdown(f"{'🟢' if HAS_GEMINI else '🔴'} **Gemini**")
    model_cols[3].markdown(f"{'🟢 Multi-model + consensus' if MULTI_MODE else '🟡 Single model' if (HAS_DEEPSEEK or HAS_OPENAI or HAS_GEMINI) else '⚪ Heuristic only'}")

    if MULTI_MODE:
        st.info("Multi-model review active. Each section is reviewed by all available models, then DeepSeek-Reasoner synthesizes the consensus. Takes 2-4 minutes for a full article.")
    elif HAS_DEEPSEEK or HAS_OPENAI or HAS_GEMINI:
        st.info("Single model review active. Add more API keys to enable multi-model consensus.")
    else:
        st.warning("Running in heuristic mode. Add API keys in Streamlit secrets to enable AI review.")

    with st.form("draft_review"):
        draft_title     = st.text_input("Draft title")
        topic           = st.text_input("Target topic / query")
        content_type    = st.selectbox("Content type", ["Blog article", "Landing page", "Thought leadership", "Research page"])
        primary_platform = st.selectbox("Primary platform", ["Google AI Overviews", "Google AI Mode", "ChatGPT Search", "Perplexity", "Multi-platform"])
        draft_text      = st.text_area("Paste article text", height=320)
        analyze         = st.form_submit_button("Review draft")

    if analyze:
        if not draft_text.strip():
            st.error("Paste article text first.")
        else:
            with st.spinner("Reviewing all sections sequentially across models... this takes 2-4 minutes for a full article."):
                result = run_review(draft_text, practices)

            # ── Overall score ──────────────────────────────────────────────
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall GEO score", f"{result['score']} / 100")
            c2.metric("Sections reviewed", len(result.get("sections", [])))
            c3.metric("Mode", result["mode"].replace("_", " ").title())

            # ── Executive summary ──────────────────────────────────────────
            if result.get("executive_summary"):
                st.divider()
                st.markdown("### Executive Summary")
                st.markdown(result["executive_summary"])

            # ── Section-by-section ─────────────────────────────────────────
            if result.get("sections"):
                st.divider()
                st.markdown("### Section-by-Section Review")

                for sec in result["sections"]:
                    label     = sec["label"]
                    original  = sec["original"]
                    avg_score = sec.get("avg_score")
                    succeeded = sec.get("models_succeeded", 0)

                    score_color   = "green" if avg_score and avg_score >= 70 else ("orange" if avg_score and avg_score >= 50 else "red")
                    score_display = f":{score_color}[avg {avg_score}/100]" if avg_score else ""
                    models_label  = f"({succeeded} model{'s' if succeeded != 1 else ''})"

                    with st.expander(f"**{label}** {score_display} {models_label}", expanded=(avg_score is not None and avg_score < 70)):

                        # Original text
                        st.markdown("**Original text**")
                        st.markdown(f"> {original}")
                        st.divider()

                        # Individual model results
                        model_tabs = []
                        model_data = []
                        for name, key in [("DeepSeek", "deepseek"), ("OpenAI", "openai"), ("Gemini", "gemini")]:
                            r = sec.get(key)
                            if r:
                                model_tabs.append(f"{name} ({r.get('geo_score', '?')}/100)")
                                model_data.append((name, r))

                        if model_data:
                            tabs = st.tabs([t for t in model_tabs] + ["✅ Consensus"])

                            for i, (name, r) in enumerate(model_data):
                                with tabs[i]:
                                    if r.get("issues"):
                                        st.markdown("**Issues found**")
                                        for issue in r["issues"]:
                                            st.markdown(f"- {issue}")
                                    if r.get("suggestion"):
                                        st.markdown("**Suggested rewrite**")
                                        st.markdown(f"> {r['suggestion']}")
                                    if r.get("why"):
                                        st.markdown(f"*Why this helps:* {r['why']}")

                            # Consensus tab — always last
                            with tabs[-1]:
                                consensus = sec.get("consensus", {})
                                if consensus.get("agreement_points"):
                                    st.markdown("**What all models agreed on**")
                                    for point in consensus["agreement_points"]:
                                        st.markdown(f"- {point}")
                                if consensus.get("consensus_suggestion"):
                                    st.markdown("**Consensus rewrite**")
                                    st.markdown(f"> {consensus['consensus_suggestion']}")
                                if consensus.get("consensus_why"):
                                    st.markdown(f"*Why:* {consensus['consensus_why']}")
                                if consensus.get("consensus_score"):
                                    st.metric("Expected GEO score after rewrite", f"{consensus['consensus_score']} / 100")

            # ── Heuristic fallback ─────────────────────────────────────────
            elif result.get("suggestions"):
                st.divider()
                st.markdown("### Suggested changes")
                for item in result["suggestions"]:
                    st.write(f"- {item}")

            save_review(draft_title or "Untitled draft", topic or "", content_type, primary_platform, result)
            st.caption("Review saved to log.")
