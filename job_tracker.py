#!/usr/bin/env python3
"""
Job Search Tracker — Streamlit app.
SQLite backend · keyword extraction · PDF resume matcher.
"""

import io
import re
import sqlite3
from collections import Counter
from datetime import datetime

import pandas as pd
import pdfplumber
import streamlit as st

DB_PATH = "jobs.db"

STATUSES = ["Interested", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"]

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","can","it","its","this","that","these","those",
    "i","you","he","she","we","they","who","which","what","as","if","then",
    "than","so","not","no","just","also","more","most","other","some","such",
    "each","every","all","any","few","much","many","own","our","your","their",
    "my","his","her","us","them","how","when","where","why","work","working",
    "team","strong","experience","ability","skills","include","including",
    "well","new","using","responsibilities","requirements","qualifications",
    "preferred","required","must","ensure","provide","support","develop",
    "across","within","while","make","take","give","get","use","help","build",
    "based","related","key","role","join","looking","seeking","opportunity",
    "position","candidate","company","please","apply","about","great","good",
    "need","both","here","there","want","know","like","time","way","business",
    "able","different","used","per","eg","ie","etc","s","t","re","ve","ll",
    "d","m","o","y","has","have","been","their","they","you","your","our",
    "us","we","he","she","it","its","this","that","these","those","who",
}


# ── Database ──────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                company     TEXT    NOT NULL,
                location    TEXT    DEFAULT '',
                url         TEXT    DEFAULT '',
                salary      TEXT    DEFAULT '',
                status      TEXT    DEFAULT 'Interested',
                description TEXT    DEFAULT '',
                notes       TEXT    DEFAULT '',
                date_added  TEXT    DEFAULT (date('now'))
            )
        """)


def add_job(title, company, location, url, salary, status, description, notes):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (title,company,location,url,salary,status,description,notes) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (title, company, location, url, salary, status, description, notes),
        )


def get_jobs(status_f="All", company_f="All", location_f="All"):
    q = "SELECT * FROM jobs WHERE 1=1"
    params = []
    if status_f != "All":
        q += " AND status=?"; params.append(status_f)
    if company_f != "All":
        q += " AND company=?"; params.append(company_f)
    if location_f != "All":
        q += " AND location=?"; params.append(location_f)
    q += " ORDER BY date_added DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params).fetchall()]


def update_job_status(job_id, status):
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))


def delete_job(job_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def get_job(job_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


# ── Keyword helpers ───────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 30) -> list:
    words = re.findall(r'\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b', text.lower())
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return Counter(filtered).most_common(top_n)


def pdf_to_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def resume_match(jd_text: str, resume_text: str):
    jd_kw   = {w for w, _ in extract_keywords(jd_text, 40)}
    res_kw  = {w for w, _ in extract_keywords(resume_text, 300)}
    matched = sorted(jd_kw & res_kw)
    missing = sorted(jd_kw - res_kw)
    score   = round(len(matched) / len(jd_kw) * 100, 1) if jd_kw else 0.0
    return score, matched, missing


# ── Status colours ────────────────────────────────────────────────────────

STATUS_COLORS = {
    "Interested":   "#5bc8f5",
    "Applied":      "#a78bfa",
    "Phone Screen": "#fbbf24",
    "Interview":    "#34d399",
    "Offer":        "#10b981",
    "Rejected":     "#f87171",
    "Withdrawn":    "#9ca3af",
}


def status_badge(status):
    color = STATUS_COLORS.get(status, "#9ca3af")
    return (
        f'<span style="background:{color}22;color:{color};'
        f'border:1px solid {color};border-radius:12px;'
        f'padding:2px 10px;font-size:0.75rem;font-weight:600;">'
        f'{status}</span>'
    )


# ── Page setup ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Job Tracker", page_icon="💼", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0f1117; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; }

  .job-card {
    background: #1e2130; border: 1px solid #2d3250;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 12px;
  }
  .job-title   { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
  .job-company { font-size: 0.95rem; color: #a0aec0; margin: 2px 0 6px; }
  .job-meta    { font-size: 0.8rem;  color: #718096; }

  .metric-card {
    background: #1e2130; border-radius: 10px;
    padding: 16px; text-align: center;
  }
  .metric-num   { font-size: 2rem; font-weight: 700; color: #ffffff; }
  .metric-label { font-size: 0.8rem; color: #718096; margin-top: 2px; }

  div[data-testid="stTabs"] button {
    font-size: 0.9rem; font-weight: 600;
  }
  div[data-testid="stButton"] > button {
    border-radius: 8px;
  }
  .stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #1e2130 !important; color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)

init_db()


# ── Tabs ──────────────────────────────────────────────────────────────────

tab_board, tab_add, tab_match = st.tabs(["📋  Job Board", "➕  Add Job", "🔍  Resume Match"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Job Board
# ══════════════════════════════════════════════════════════════════════════

with tab_board:
    st.markdown("## 💼 Job Board")

    all_jobs = get_jobs()

    if not all_jobs:
        st.info("No jobs saved yet. Go to **Add Job** to get started.")
    else:
        # ── Summary metrics ──────────────────────────────────────────────
        status_counts = Counter(j["status"] for j in all_jobs)
        cols = st.columns(len(STATUSES))
        for col, s in zip(cols, STATUSES):
            with col:
                color = STATUS_COLORS.get(s, "#9ca3af")
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-num" style="color:{color}">{status_counts.get(s,0)}</div>'
                    f'<div class="metric-label">{s}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filters ──────────────────────────────────────────────────────
        companies  = ["All"] + sorted({j["company"]  for j in all_jobs if j["company"]})
        locations  = ["All"] + sorted({j["location"] for j in all_jobs if j["location"]})

        fc, fl, fs = st.columns(3)
        with fc:
            company_f  = st.selectbox("Company",  companies,  key="bf_company")
        with fl:
            location_f = st.selectbox("Location", locations,  key="bf_location")
        with fs:
            status_f   = st.selectbox("Status",   ["All"] + STATUSES, key="bf_status")

        filtered = get_jobs(status_f, company_f, location_f)

        st.markdown(f"**{len(filtered)} job{'s' if len(filtered)!=1 else ''}**")
        st.markdown("---")

        # ── Job cards ────────────────────────────────────────────────────
        for job in filtered:
            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    title_html = (
                        f'<a href="{job["url"]}" target="_blank" '
                        f'style="color:#ffffff;text-decoration:none;">{job["title"]}</a>'
                        if job["url"] else job["title"]
                    )
                    salary_str = f' · 💰 {job["salary"]}' if job["salary"] else ""
                    location_str = f' · 📍 {job["location"]}' if job["location"] else ""
                    st.markdown(
                        f'<div class="job-card">'
                        f'<div class="job-title">{title_html}</div>'
                        f'<div class="job-company">🏢 {job["company"]}</div>'
                        f'<div style="margin-bottom:8px;">{status_badge(job["status"])}</div>'
                        f'<div class="job-meta">{job["date_added"]}{location_str}{salary_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    new_status = st.selectbox(
                        "Status", STATUSES,
                        index=STATUSES.index(job["status"]) if job["status"] in STATUSES else 0,
                        key=f"status_{job['id']}",
                        label_visibility="collapsed",
                    )
                    if new_status != job["status"]:
                        update_job_status(job["id"], new_status)
                        st.rerun()

                    if st.button("🗑 Delete", key=f"del_{job['id']}"):
                        delete_job(job["id"])
                        st.rerun()

                # Expandable details
                with st.expander("View description & notes"):
                    if job["description"]:
                        st.markdown("**Job Description**")
                        st.text(job["description"])
                        st.markdown("**Top Keywords**")
                        kw = extract_keywords(job["description"], 15)
                        if kw:
                            st.markdown(" · ".join(
                                f'`{w}` ×{c}' for w, c in kw
                            ))
                    if job["notes"]:
                        st.markdown("**Notes**")
                        st.info(job["notes"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Add Job
# ══════════════════════════════════════════════════════════════════════════

with tab_add:
    st.markdown("## ➕ Add a Job")

    with st.form("add_job_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            title   = st.text_input("Job Title *", placeholder="e.g. Senior Product Manager")
            company = st.text_input("Company *",   placeholder="e.g. Acme Corp")
            location = st.text_input("Location",   placeholder="e.g. Seattle, WA / Remote")
        with c2:
            url    = st.text_input("Job URL",      placeholder="https://...")
            salary = st.text_input("Salary Range", placeholder="e.g. $120k–$150k")
            status = st.selectbox("Status", STATUSES)

        description = st.text_area(
            "Job Description",
            placeholder="Paste the full job description here for keyword analysis…",
            height=200,
        )
        notes = st.text_area("Your Notes", placeholder="Why you're interested, referral contact, etc.", height=80)

        submitted = st.form_submit_button("💾 Save Job", use_container_width=True)

    if submitted:
        if not title or not company:
            st.error("Job Title and Company are required.")
        else:
            add_job(title, company, location, url, salary, status, description, notes)
            st.success(f"✅ **{title}** at **{company}** saved!")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Resume Match
# ══════════════════════════════════════════════════════════════════════════

with tab_match:
    st.markdown("## 🔍 Resume Keyword Matcher")

    all_jobs = get_jobs()
    jobs_with_jd = [j for j in all_jobs if j["description"] and j["description"].strip()]

    if not jobs_with_jd:
        st.info("Add jobs with a job description first, then come back here to match your resume.")
    else:
        c1, c2 = st.columns([3, 2])

        with c1:
            job_options = {f'{j["title"]} — {j["company"]}': j["id"] for j in jobs_with_jd}
            selected_label = st.selectbox("Select a job to match against", list(job_options.keys()))
            selected_id = job_options[selected_label]
            job = get_job(selected_id)

            resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

            if resume_file and job:
                resume_text = pdf_to_text(resume_file.read())

                if not resume_text.strip():
                    st.error("Could not extract text from the PDF. Try a text-based PDF (not a scanned image).")
                else:
                    score, matched, missing = resume_match(job["description"], resume_text)

                    # Score gauge
                    color = "#10b981" if score >= 70 else "#fbbf24" if score >= 40 else "#f87171"
                    st.markdown(
                        f'<div style="background:#1e2130;border-radius:12px;padding:24px;text-align:center;margin:16px 0;">'
                        f'<div style="font-size:3.5rem;font-weight:700;color:{color}">{score}%</div>'
                        f'<div style="color:#718096;font-size:0.9rem">keyword match score</div>'
                        f'<div style="background:#2d3250;border-radius:8px;height:10px;margin-top:12px;">'
                        f'<div style="background:{color};width:{score}%;height:10px;border-radius:8px;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                    mc, gc = st.columns(2)
                    with mc:
                        st.markdown("**✅ Keywords Found in Resume**")
                        if matched:
                            st.markdown(" ".join(f'`{w}`' for w in matched))
                        else:
                            st.markdown("_None matched_")
                    with gc:
                        st.markdown("**❌ Missing Keywords — Add to Resume**")
                        if missing:
                            st.markdown(" ".join(f'`{w}`' for w in missing))
                        else:
                            st.markdown("_Great — no gaps!_")

        with c2:
            if job:
                st.markdown("**Job Description Keywords**")
                kw = extract_keywords(job["description"], 20)
                for word, count in kw:
                    pct = count / kw[0][1] if kw else 0
                    bar_w = int(pct * 100)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                        f'<span style="min-width:130px;font-size:0.85rem;color:#e2e8f0">{word}</span>'
                        f'<div style="flex:1;background:#2d3250;border-radius:4px;height:8px;">'
                        f'<div style="background:#5bc8f5;width:{bar_w}%;height:8px;border-radius:4px;"></div>'
                        f'</div>'
                        f'<span style="font-size:0.75rem;color:#718096;min-width:20px">{count}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
