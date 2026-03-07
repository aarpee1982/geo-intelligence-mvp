from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    BEST_PRACTICES_CSV, DISCOVERED_CSV, MANUAL_SUBMISSIONS_CSV,
    REVIEWS_CSV, SOURCES_CSV,
    ensure_manual_submission_table, ensure_reviews_table,
    load_csv, next_id, now_iso, save_csv, url_hash, workbook_bytes,
)

st.set_page_config(page_title="GEO Intelligence Hub", layout="wide")
st.title("GEO Intelligence Hub")
st.caption("Source ingestion, best-practice library, and article reviewer for AI search.")

# ── Model availability (functions so env is read fresh each call) ─────────────
def HAS_DEEPSEEK(): return bool(os.getenv("DEEPSEEK_API_KEY"))
def HAS_KIMI():     return bool(os.getenv("KIMI_API_KEY"))
def MULTI_MODE():   return HAS_DEEPSEEK() and HAS_KIMI()

# ── Auto-approve ──────────────────────────────────────────────────────────────
AUTO_APPROVE_SCORE = 60
AUTO_APPROVE_CATEGORIES = {"official", "research", "vendor_research", "practitioner", "analyst"}

def promote_to_library(row, approved_df, source="auto"):
    existing_urls = set(approved_df.get("URL", pd.Series(dtype=str)).astype(str).tolist())
    if row.get("URL", "") in existing_urls:
        return approved_df
    new_id = next_id("SRC", approved_df.get("Source ID", pd.Series(dtype=str)))
    new_row = {
        "Source ID": new_id, "Title": row.get("Title", ""), "URL": row.get("URL", ""),
        "Source Type": "Auto-approved / pending human review" if source == "auto" else "Boss submission",
        "Publisher": row.get("Feed Name", ""), "Author": "",
        "Publication Date": row.get("Published Date", ""),
        "Platform Relevance": row.get("Platform Relevance", ""),
        "Evidence Type": "Auto-discovered", "Sample Size / Scope": "",
        "Core Claim": str(row.get("Summary", "")),
        "Extracted Findings": row.get("Summary", ""),
        "Operational Implications": "", "Confidence Level": "Emerging",
        "Contradictions / Caveats": "", "Tags": row.get("Feed Category", ""),
        "Last Reviewed": now_iso()[:10],
    }
    return pd.concat([approved_df, pd.DataFrame([new_row])], ignore_index=True)

def bulk_auto_approve(discovered):
    approved_df = load_csv(SOURCES_CSV)
    candidates = discovered[discovered["Status"] == "candidate"].copy() if "Status" in discovered.columns else discovered.copy()
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

def auto_approve_boss_submissions(submissions):
    approved_df = load_csv(SOURCES_CSV)
    new_subs = submissions[submissions.get("Status", pd.Series(dtype=str)) == "new"]
    count = 0
    for _, row in new_subs.iterrows():
        fake_row = pd.Series({
            "Title": row.get("Title / Note", ""), "URL": row.get("URL", ""),
            "Feed Name": "Boss submission by " + row.get("Submitted By", "Unknown"),
            "Feed Category": "boss_submission",
            "Published Date": row.get("Submitted At", "")[:10],
            "Platform Relevance": "", "Summary": row.get("Why It Matters", ""),
            "Authority Score": 100,
        })
        approved_df = promote_to_library(fake_row, approved_df, source="boss")
        submissions.loc[submissions["Submission ID"] == row["Submission ID"], "Status"] = "approved"
        count += 1
    return approved_df, submissions, count

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_rules_summary(practices):
    if practices.empty:
        return ""
    return "\n".join(
        "- " + str(r["Rule Title"]) + ": " + str(r["Rule Statement"])
        for _, r in practices.head(10).iterrows()
        if pd.notna(r.get("Rule Title")) and pd.notna(r.get("Rule Statement"))
    )

def build_prompt(label, text, rules):
    return (
        "You are a GEO editor. Review this article section so it gets cited by AI search engines.\n\n"
        "GEO best practices:\n" + rules + "\n\n"
        "Section: " + label + "\n"
        "Text: " + text + "\n\n"
        "Return ONLY valid JSON with keys:\n"
        "geo_score (int 0-100), issues (list of strings), suggestion (rewritten text), why (one sentence)"
    )

# ── API clients ───────────────────────────────────────────────────────────────
def deepseek_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def kimi_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("KIMI_API_KEY"), base_url="https://api.moonshot.cn/v1")

# ── Section reviewers ─────────────────────────────────────────────────────────
def review_deepseek(label, text, rules):
    try:
        r = deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": build_prompt(label, text, rules)}],
            max_tokens=800, temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        st.warning("DeepSeek error on " + label + ": " + str(e))
        return None

def review_kimi(label, text, rules):
    try:
        r = kimi_client().chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": build_prompt(label, text, rules)}],
            max_tokens=800, temperature=0.3,
        )
        raw = r.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        st.warning("Kimi error on " + label + ": " + str(e))
        return None

# ── Consensus ─────────────────────────────────────────────────────────────────
def synthesize_consensus(label, text, ds, kimi, rules):
    parts = []
    if ds:
        parts.append("DeepSeek: issues=" + str(ds.get("issues")) + " suggestion=" + str(ds.get("suggestion")))
    if kimi:
        parts.append("Kimi: issues=" + str(kimi.get("issues")) + " suggestion=" + str(kimi.get("suggestion")))
    if not parts:
        return {"consensus_suggestion": text, "consensus_why": "No models returned results.", "consensus_score": None, "agreement_points": []}
    if len(parts) == 1:
        result = ds or kimi
        return {"consensus_suggestion": result.get("suggestion", text), "consensus_why": result.get("why", ""), "consensus_score": result.get("geo_score"), "agreement_points": []}
    prompt = (
        "You are DeepSeek-Reasoner. Synthesize GEO feedback into one consensus rewrite.\n"
        "Section: " + label + "\nOriginal: " + text + "\n"
        "Reviews:\n" + "\n".join(parts) + "\n"
        "Return ONLY valid JSON: consensus_suggestion, consensus_why (2 sentences), consensus_score (0-100), agreement_points (list of 1-3 strings)"
    )
    for model in ["deepseek-reasoner", "deepseek-chat"]:
        try:
            r = deepseek_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000, temperature=0.2,
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content)
        except Exception:
            continue
    result = ds or kimi or {}
    return {"consensus_suggestion": result.get("suggestion", text), "consensus_why": "Consensus unavailable.", "consensus_score": None, "agreement_points": []}

# ── Executive summary ─────────────────────────────────────────────────────────
def executive_summary(section_results, rules):
    try:
        scores = [r.get("avg_score") for r in section_results if r.get("avg_score") is not None]
        avg = round(sum(scores) / len(scores)) if scores else "N/A"
        issues = []
        for r in section_results:
            for k in ["deepseek", "kimi"]:
                for issue in (r.get(k) or {}).get("issues", []):
                    issues.append(issue)
        prompt = (
            "You are a senior GEO strategist. Write a 3-5 sentence executive summary of this article's AI-citation readiness.\n"
            "Average GEO score: " + str(avg) + "/100. Sections reviewed: " + str(len(section_results)) + ".\n"
            "Key issues: " + json.dumps(issues[:10]) + "\n"
            "Be direct. State the verdict, name 2-3 fixes, end with what the article achieves after those fixes. No bullet points."
        )
        r = deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.3,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return "Executive summary unavailable: " + str(e)

# ── Heuristic fallback ────────────────────────────────────────────────────────
def heuristic_review(text, practices):
    text_low = text.lower()
    words = re.findall(r"\w+", text)
    first_120 = " ".join(words[:120]).lower()
    violated, suggestions = [], []
    if len(first_120.split()) > 20 and not any(k in first_120 for k in ["in short", "bottom line", "the answer", "summary"]):
        violated.append("BP-01"); suggestions.append("Add a direct answer in the opening 2-4 lines.")
    if any(len(s.split()) > 28 for s in re.split(r"[.!?]\s+", text)):
        violated.append("BP-02"); suggestions.append("Split long sentences into shorter factual claims.")
    if not any(m in text_low for m in ["##", "faq", "table", "key takeaways"]):
        violated.append("BP-03"); suggestions.append("Add subheads, FAQ blocks, or tables.")
    if not any(m in text_low for m in ["source", "study", "data", "according to"]):
        violated.append("BP-06"); suggestions.append("Anchor claims in explicit evidence.")
    score = max(35, min(100, 100 - 8 * len(set(violated))))
    return {"matched": [], "violated": sorted(set(violated)), "suggestions": suggestions,
            "score": score, "rationale": [], "sections": [], "executive_summary": "", "mode": "heuristic"}

# ── Main review ───────────────────────────────────────────────────────────────
def run_review(text, practices):
    if not HAS_DEEPSEEK() and not HAS_KIMI():
        return heuristic_review(text, practices)

    rules = get_rules_summary(practices)
    words = text.split()
    chunk_size = 100
    sections = []
    for i in range(0, min(len(words), 400), chunk_size):
        chunk = words[i:i+chunk_size]
        if chunk:
            sections.append({
                "label": "Section " + str(len(sections)+1) + " (words " + str(i+1) + "-" + str(i+len(chunk)) + ")",
                "text": " ".join(chunk)
            })

    results = []
    for sec in sections:
        ds   = review_deepseek(sec["label"], sec["text"], rules) if HAS_DEEPSEEK() else None
        kimi = review_kimi(sec["label"], sec["text"], rules)     if HAS_KIMI()     else None
        consensus = synthesize_consensus(sec["label"], sec["text"], ds, kimi, rules)
        scores = [r.get("geo_score") for r in [ds, kimi] if r and r.get("geo_score") is not None]
        results.append({
            "label": sec["label"], "original": sec["text"],
            "deepseek": ds, "kimi": kimi,
            "consensus": consensus,
            "avg_score": round(sum(scores)/len(scores)) if scores else None,
            "models_succeeded": sum(1 for r in [ds, kimi] if r is not None),
        })

    exec_sum = executive_summary(results, rules) if HAS_DEEPSEEK() else ""
    scores = [r["avg_score"] for r in results if r.get("avg_score") is not None]
    overall = round(sum(scores)/len(scores)) if scores else 50
    mode = "multi_model" if MULTI_MODE() else "single_model"

    return {"score": overall, "sections": results, "executive_summary": exec_sum,
            "matched": [], "violated": [], "suggestions": [], "rationale": [], "mode": mode}

# ── Save helpers ──────────────────────────────────────────────────────────────
def save_submission(submitter, url, title, why, priority):
    df = ensure_manual_submission_table()
    dedup = url_hash(url)
    if not df.empty and dedup in df.get("Dedup Key", pd.Series(dtype=str)).astype(str).tolist():
        st.warning("This link already exists in the intake queue.")
        return
    sid = next_id("SUB", df.get("Submission ID", pd.Series(dtype=str)))
    row = {"Submission ID": sid, "Submitted At": now_iso(), "Submitted By": submitter,
           "URL": url, "Title / Note": title, "Why It Matters": why,
           "Priority": priority, "Status": "new", "Dedup Key": dedup}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, MANUAL_SUBMISSIONS_CSV)
    approved_df, _, _ = auto_approve_boss_submissions(df)
    save_csv(approved_df, SOURCES_CSV)
    save_csv(df, MANUAL_SUBMISSIONS_CSV)
    st.success("Saved " + sid + " and auto-approved into the Source Library.")

def save_review(title, topic, content_type, platform, result):
    df = ensure_reviews_table()
    rid = next_id("REV", df.get("Review ID", pd.Series(dtype=str)))
    row = {"Review ID": rid, "Draft Title": title, "Target Query / Topic": topic,
           "Content Type": content_type, "Primary Platform": platform,
           "Matched Best Practices": "", "Violated Best Practices": "; ".join(result.get("violated", [])),
           "Suggested Edits": " | ".join(result.get("suggestions", [])),
           "Applied Edits": "", "What Changed": "", "Why It Changed": "",
           "Evidence Links Used": "", "Human Override Notes": "",
           "Final Score / 100": result.get("score", ""), "Reviewed Date": now_iso()[:10]}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, REVIEWS_CSV)

# ── Data ──────────────────────────────────────────────────────────────────────
sources     = load_csv(SOURCES_CSV)
practices   = load_csv(BEST_PRACTICES_CSV)
submissions = ensure_manual_submission_table()
reviews     = ensure_reviews_table()
discovered  = load_csv(DISCOVERED_CSV)

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

# Tab 1 ────────────────────────────────────────────────────────────────────────
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

# Tab 2 ────────────────────────────────────────────────────────────────────────
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

# Tab 3 ────────────────────────────────────────────────────────────────────────
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
                st.markdown("**" + str(eligible_count) + " candidates** eligible for bulk auto-approve (score 60+).")
            with col2:
                if st.button("Bulk auto-approve " + str(eligible_count) + " candidates", type="primary"):
                    approved_df, updated_discovered, count = bulk_auto_approve(discovered)
                    save_csv(approved_df, SOURCES_CSV)
                    save_csv(updated_discovered, DISCOVERED_CSV)
                    st.success("Auto-approved " + str(count) + " sources.")
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
                    st.write("**" + str(selected_row.get("Title", "")) + "**")
                    st.write(str(selected_row.get("Summary", "")))
                    source_type = st.selectbox("Source type", ["Study / Article", "Vendor study / Article", "Official documentation", "Research paper", "Industry report"])
                    confidence  = st.selectbox("Confidence level", ["Strong", "Official", "Emerging", "Speculative"])
                    core_claim  = st.text_area("Core claim", value=str(selected_row.get("Summary", "")))
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
                            "Core Claim": core_claim, "Extracted Findings": str(selected_row.get("Summary", "")),
                            "Operational Implications": "", "Confidence Level": confidence,
                            "Contradictions / Caveats": "", "Tags": tags, "Last Reviewed": now_iso()[:10],
                        }
                        approved_df = pd.concat([approved_df, pd.DataFrame([new_row])], ignore_index=True)
                        save_csv(approved_df, SOURCES_CSV)
                        discovered.loc[discovered["URL"] == selected_url, "Status"] = "approved"
                        save_csv(discovered, DISCOVERED_CSV)
                        st.success("Approved as " + new_id + ".")
                        st.rerun()

# Tab 4 ────────────────────────────────────────────────────────────────────────
with practice_tab:
    st.subheader("Canonical best-practice library")
    confidence_opts = sorted(practices["Confidence Level"].dropna().unique().tolist()) if not practices.empty else []
    confidence = st.multiselect("Filter by confidence", confidence_opts, default=confidence_opts)
    pview = practices[practices["Confidence Level"].isin(confidence)] if confidence else practices
    st.dataframe(pview, use_container_width=True, hide_index=True)

# Tab 5 ────────────────────────────────────────────────────────────────────────
with review_tab:
    st.subheader("Article reviewer")

    mc = st.columns(3)
    mc[0].markdown(("🟢" if HAS_DEEPSEEK() else "🔴") + " **DeepSeek**")
    mc[1].markdown(("🟢" if HAS_KIMI() else "🔴") + " **Kimi**")
    mc[2].markdown("🟢 Multi-model + consensus" if MULTI_MODE() else ("🟡 Single model" if HAS_DEEPSEEK() or HAS_KIMI() else "⚪ Heuristic only"))

    if MULTI_MODE():
        st.info("Multi-model review active. DeepSeek and Kimi each review every section, then DeepSeek-Reasoner synthesizes the consensus.")
    elif HAS_DEEPSEEK() or HAS_KIMI():
        st.info("Single model active. Add both DEEPSEEK_API_KEY and KIMI_API_KEY in Streamlit secrets for full consensus mode.")
    else:
        st.warning("Heuristic mode only. Add API keys in Streamlit Cloud secrets to enable AI review.")

    with st.form("draft_review"):
        draft_title      = st.text_input("Draft title")
        topic            = st.text_input("Target topic / query")
        content_type     = st.selectbox("Content type", ["Blog article", "Landing page", "Thought leadership", "Research page"])
        primary_platform = st.selectbox("Primary platform", ["Google AI Overviews", "Google AI Mode", "ChatGPT Search", "Perplexity", "Multi-platform"])
        draft_text       = st.text_area("Paste article text", height=320)
        analyze          = st.form_submit_button("Review draft")

    if analyze:
        if not draft_text.strip():
            st.error("Paste article text first.")
        else:
            with st.spinner("Reviewing first 400 words across all models... takes 1-3 minutes."):
                result = run_review(draft_text, practices)

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall GEO score", str(result["score"]) + " / 100")
            c2.metric("Sections reviewed", len(result.get("sections", [])))
            c3.metric("Mode", result["mode"].replace("_", " ").title())

            if result.get("executive_summary"):
                st.divider()
                st.markdown("### Executive Summary")
                st.markdown(result["executive_summary"])

            if result.get("sections"):
                st.divider()
                st.markdown("### Section-by-Section Review")
                for sec in result["sections"]:
                    avg   = sec.get("avg_score")
                    succ  = sec.get("models_succeeded", 0)
                    color = "green" if avg and avg >= 70 else ("orange" if avg and avg >= 50 else "red")
                    score_badge = (":" + color + "[avg " + str(avg) + "/100]") if avg else ""
                    with st.expander("**" + sec["label"] + "** " + score_badge + " (" + str(succ) + " model" + ("s" if succ != 1 else "") + ")", expanded=(avg is not None and avg < 70)):
                        st.markdown("**Original text**")
                        st.markdown("> " + sec["original"])
                        st.divider()

                        model_data = [(n, k) for n, k in [("DeepSeek", "deepseek"), ("Kimi", "kimi")] if sec.get(k)]
                        if model_data:
                            tab_labels = [n + " (" + str(sec[k].get("geo_score", "?")) + "/100)" for n, k in model_data] + ["✅ Consensus"]
                            tabs = st.tabs(tab_labels)
                            for i, (name, key) in enumerate(model_data):
                                r = sec[key]
                                with tabs[i]:
                                    if r.get("issues"):
                                        st.markdown("**Issues found**")
                                        for issue in r["issues"]:
                                            st.markdown("- " + str(issue))
                                    if r.get("suggestion"):
                                        st.markdown("**Suggested rewrite**")
                                        st.markdown("> " + str(r["suggestion"]))
                                    if r.get("why"):
                                        st.markdown("*Why this helps:* " + str(r["why"]))
                            with tabs[-1]:
                                con = sec.get("consensus", {})
                                if con.get("agreement_points"):
                                    st.markdown("**What both models agreed on**")
                                    for pt in con["agreement_points"]:
                                        st.markdown("- " + str(pt))
                                if con.get("consensus_suggestion"):
                                    st.markdown("**Consensus rewrite**")
                                    st.markdown("> " + str(con["consensus_suggestion"]))
                                if con.get("consensus_why"):
                                    st.markdown("*Why:* " + str(con["consensus_why"]))
                                if con.get("consensus_score"):
                                    st.metric("Expected GEO score after rewrite", str(con["consensus_score"]) + " / 100")
                        else:
                            st.warning("No model results for this section.")

            elif result.get("suggestions"):
                st.divider()
                st.markdown("### Suggested changes")
                for item in result["suggestions"]:
                    st.write("- " + item)

            save_review(draft_title or "Untitled draft", topic or "", content_type, primary_platform, result)
            st.caption("Review saved to log.")
