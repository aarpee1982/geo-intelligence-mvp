from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    BEST_PRACTICES_CSV,
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
st.caption("Boss-friendly intake, evolving best-practice library, and article review on top of your seeded research base.")


def score_draft(text: str, practices: pd.DataFrame) -> dict:
    text_low = text.lower()
    words = re.findall(r"\w+", text)
    first_120 = " ".join(words[:120]).lower()

    matched = []
    violated = []
    suggestions = []

    if len(words) > 0:
        if len(first_120.split()) > 20 and not any(k in first_120 for k in ["in short", "bottom line", "key takeaway", "the answer", "summary"]):
            violated.append("BP-01")
            suggestions.append("Add a direct answer block in the opening 2–4 lines.")
        else:
            matched.append("BP-01")

    long_sentences = [s for s in re.split(r"[.!?]\s+", text) if len(s.split()) > 28]
    if long_sentences:
        violated.append("BP-02")
        suggestions.append("Split long compound sentences into shorter stand-alone factual claims.")
    else:
        matched.append("BP-02")

    if not any(marker in text_low for marker in ["##", "###", "faq", "table", "key takeaways", "question"]):
        violated.append("BP-03")
        suggestions.append("Add stronger structure such as subheads, FAQ blocks, or comparison tables.")
    else:
        matched.append("BP-03")

    if not any(marker in text_low for marker in ["related", "also", "adjacent", "compare", "versus", "faq", "questions"]):
        violated.append("BP-04")
        suggestions.append("Expand adjacent-question coverage so the page is useful beyond the head term.")
    else:
        matched.append("BP-04")

    if not any(marker in text_low for marker in ["source", "study", "data", "according to", "official"]):
        violated.append("BP-06")
        suggestions.append("Anchor claims in explicit evidence rather than generic assertions.")
    else:
        matched.append("BP-06")

    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    score = 100
    score -= 8 * len(set(violated))
    if unique_ratio < 0.42:
        score -= 8
        suggestions.append("The draft may be too repetitive. Tighten wording and remove filler.")
    score = max(35, min(100, score))

    rationale = []
    for bp in sorted(set(violated)):
        row = practices[practices["Practice ID"] == bp]
        if not row.empty:
            r = row.iloc[0]
            rationale.append(f"{bp} — {r['Rule Title']}: {r['Why It Matters']}")

    return {
        "matched": sorted(set(matched)),
        "violated": sorted(set(violated)),
        "suggestions": suggestions,
        "score": score,
        "rationale": rationale,
    }


def save_submission(submitter: str, url: str, title: str, why: str, priority: str) -> None:
    df = ensure_manual_submission_table()
    dedup = url_hash(url)
    if not df.empty and dedup in df.get("Dedup Key", []).astype(str).tolist():
        st.warning("This link already exists in the intake queue.")
        return
    submission_id = next_id("SUB", df.get("Submission ID", []))
    row = {
        "Submission ID": submission_id,
        "Submitted At": now_iso(),
        "Submitted By": submitter,
        "URL": url,
        "Title / Note": title,
        "Why It Matters": why,
        "Priority": priority,
        "Status": "new",
        "Dedup Key": dedup,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, MANUAL_SUBMISSIONS_CSV)
    st.success(f"Saved {submission_id} to the intake queue.")


def save_review(title: str, topic: str, content_type: str, primary_platform: str, result: dict) -> None:
    df = ensure_reviews_table()
    review_id = next_id("REV", df.get("Review ID", []))
    row = {
        "Review ID": review_id,
        "Draft Title": title,
        "Target Query / Topic": topic,
        "Content Type": content_type,
        "Primary Platform": primary_platform,
        "Matched Best Practices": "; ".join(result["matched"]),
        "Violated Best Practices": "; ".join(result["violated"]),
        "Suggested Edits": " | ".join(result["suggestions"]),
        "Applied Edits": "",
        "What Changed": "",
        "Why It Changed": " | ".join(result["rationale"]),
        "Evidence Links Used": "",
        "Human Override Notes": "",
        "Final Score / 100": result["score"],
        "Reviewed Date": now_iso()[:10],
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_csv(df, REVIEWS_CSV)
    st.success(f"Saved review {review_id}.")


sources = load_csv(SOURCES_CSV)
practices = load_csv(BEST_PRACTICES_CSV)
submissions = ensure_manual_submission_table()
reviews = ensure_reviews_table()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Seeded sources", len(sources))
k2.metric("Best practices", len(practices))
k3.metric("Manual submissions", len(submissions))
k4.metric("Reviews logged", len(reviews))

intake_tab, library_tab, practice_tab, review_tab = st.tabs([
    "Add Source",
    "Source Library",
    "Best Practices",
    "Article Reviewer",
])

with intake_tab:
    st.subheader("Boss-friendly source intake")
    with st.form("source_intake"):
        submitter = st.text_input("Submitted by", placeholder="Rahul or Boss")
        url = st.text_input("Source URL", placeholder="Paste the article, study, or PDF link")
        title = st.text_input("Title or note", placeholder="Why this may matter")
        why = st.text_area("Why it matters", placeholder="One or two lines on why this is important")
        priority = st.selectbox("Priority", ["low", "medium", "high"])
        submitted = st.form_submit_button("Save source")
        if submitted:
            if not url.strip():
                st.error("URL is required.")
            else:
                save_submission(submitter.strip() or "Unknown", url.strip(), title.strip(), why.strip(), priority)

    st.divider()
    st.subheader("Current intake queue")
    st.dataframe(submissions, use_container_width=True, hide_index=True)

with library_tab:
    st.subheader("Seeded evidence library")
    search = st.text_input("Filter by title, publisher, tag, or platform", placeholder="Google AI Mode, Ahrefs, OpenAI, citations")
    view = sources.copy()
    if search.strip():
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False))
        view = view[mask.any(axis=1)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download live workbook",
        data=workbook_bytes(),
        file_name="geo_intelligence_live.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with practice_tab:
    st.subheader("Canonical best-practice library")
    confidence = st.multiselect(
        "Filter by confidence",
        sorted(practices["Confidence Level"].dropna().unique().tolist()),
        default=sorted(practices["Confidence Level"].dropna().unique().tolist()),
    )
    pview = practices[practices["Confidence Level"].isin(confidence)] if confidence else practices
    st.dataframe(pview, use_container_width=True, hide_index=True)

with review_tab:
    st.subheader("Article reviewer")
    with st.form("draft_review"):
        draft_title = st.text_input("Draft title")
        topic = st.text_input("Target topic / query")
        content_type = st.selectbox("Content type", ["Blog article", "Landing page", "Thought leadership", "Research page"])
        primary_platform = st.selectbox("Primary platform", ["Google AI Overviews", "Google AI Mode", "ChatGPT Search", "Multi-platform"])
        draft_text = st.text_area("Paste article text", height=320)
        analyze = st.form_submit_button("Analyze draft")

    if analyze:
        if not draft_text.strip():
            st.error("Paste article text first.")
        else:
            result = score_draft(draft_text, practices)
            c1, c2, c3 = st.columns(3)
            c1.metric("Draft score", result["score"])
            c2.metric("Matched rules", len(result["matched"]))
            c3.metric("Violated rules", len(result["violated"]))

            st.markdown("### Suggested changes")
            for item in result["suggestions"]:
                st.write(f"- {item}")

            st.markdown("### Why these changes")
            for item in result["rationale"]:
                st.write(f"- {item}")

            save_review(draft_title or "Untitled draft", topic or "", content_type, primary_platform, result)
