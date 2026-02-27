#!/usr/bin/env python3
"""
Job Search Tracker — Streamlit app.
SQLite backend · keyword extraction · PDF resume matcher.
"""

import html as _html
import io
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import requests
import streamlit as st
from bs4 import BeautifulSoup
from wordcloud import WordCloud

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


# ── Description formatter ─────────────────────────────────────────────────

_BULLET_CHAR_RE = re.compile(r'^[•·▪▸►●○◦‣⁃]\s*')

_HEADER_RE = re.compile(
    r"^(?:responsibilities|requirements|qualifications|"
    r"about\s+(?:the\s+)?(?:role|team|company|position|us|this\s+role)|"
    r"what\s+you(?:ll|will|'ll)?\s+(?:do|bring|need|build)|"
    r"who\s+you\s+are|what\s+we(?:re|'re)?\s+looking\s+for|"
    r"preferred\s+qualifications?|basic\s+qualifications?|"
    r"minimum\s+qualifications?|required\s+qualifications?|"
    r"key\s+responsibilities?|additional\s+(?:or\s+)?preferred|"
    r"job\s+(?:summary|description|duties|overview)|"
    r"benefits?(?:\s+include)?|compensation(?:\s+and\s+benefits)?|"
    r"skills?(?:\s+and\s+experience)?|experience|education|"
    r"overview|summary|role\s+overview|position\s+overview|"
    r"perks?|culture|values?|equal\s+opportunity"
    r")s?:?\s*$",
    re.IGNORECASE,
)

# Matches a section boundary: text after sentence-end before a Capitalized phrase with colon
# Handles single-word ("Requirements:") and multi-word ("Go-to-Market Execution:") headers
_INLINE_SECTION_RE = re.compile(
    r'(?<=[.!?])\s{1,3}(?=[A-Z][A-Za-z&/\-]*(?:\s+[A-Za-z&/\-]+){0,4}\s*:)'
)


def _inject_structure(text: str) -> str:
    """
    Pre-process plain-text descriptions that have no newlines.
    Injects newlines before section headers (Capitalized Phrase: content)
    so that format_description_md can detect them line-by-line.
    """
    if '\n' in text:
        return text  # already has structure

    # Step 1: split BEFORE section headers that follow a sentence end
    #   e.g. "...levers. Go-to-Market Execution: Partner..." →
    #        "...levers.\n\nGo-to-Market Execution: Partner..."
    text = _INLINE_SECTION_RE.sub('\n\n', text)

    # Step 2: split the inline "Header: content" into two lines
    #   e.g. "Go-to-Market Execution: Partner with..." →
    #        "Go-to-Market Execution:\nPartner with..."
    text = re.sub(
        r'^([A-Z][A-Za-z&/ \-]{2,60}):\s+',
        r'\1:\n',
        text,
        flags=re.MULTILINE,
    )

    # Step 3: split double-spaces (often used as list-item separators in copied text)
    text = re.sub(r'  +', '\n', text)

    return text


def format_description_md(text: str) -> str:
    """
    Convert raw scraped job description text into readable markdown.
    Handles plain-text (no newlines), unicode bullets, and HTML entities.
    """
    text = _html.unescape(text)
    text = _inject_structure(text)
    lines = text.split('\n')

    # Build a list of "blocks" — each block becomes a separate markdown paragraph.
    # Bullets are accumulated into a single block (compact list).
    # Regular text lines are accumulated into a single paragraph block.
    blocks = []  # each item: str (header/paragraph) or list[str] (bullet group)

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Unicode bullet symbols → markdown list items
        if _BULLET_CHAR_RE.match(s):
            item = '- ' + _BULLET_CHAR_RE.sub('', s).strip()
            if blocks and isinstance(blocks[-1], list):
                blocks[-1].append(item)
            else:
                blocks.append([item])

        # Existing dash/star bullet or numbered list
        elif re.match(r'^[-*]\s+\S', s) or re.match(r'^\d+[.)]\s+\S', s):
            if blocks and isinstance(blocks[-1], list):
                blocks[-1].append(s)
            else:
                blocks.append([s])

        # Known section header keyword
        elif _HEADER_RE.match(s):
            blocks.append(f'**{s.rstrip(":")}**')

        # Short line ending with ":" → inferred header
        elif (s.endswith(':') and len(s) < 80
              and not re.search(r'[,;]', s)
              and len(s.split()) <= 10):
            blocks.append(f'**{s.rstrip(":")}**')

        # ALL CAPS short line → inferred header
        elif s.isupper() and 3 < len(s) < 80 and len(s.split()) <= 8:
            blocks.append(f'**{s.title()}**')

        else:
            # Regular text: accumulate into previous paragraph if possible
            if blocks and isinstance(blocks[-1], str) and not blocks[-1].startswith('**'):
                blocks[-1] = blocks[-1] + ' ' + s
            else:
                blocks.append(s)

    # Render blocks: lists → compact bullet markdown, strings → as-is
    parts = []
    for b in blocks:
        if isinstance(b, list):
            parts.append('\n'.join(b))
        else:
            parts.append(b)

    return '\n\n'.join(parts)


# ── Salary text extractor ─────────────────────────────────────────────────

# Keywords that signal a salary sentence is nearby
_SAL_TRIGGER_RE = re.compile(
    r'(?:'
    r'base\s+pay\s+range'
    r'|base\s+salary\s+range'
    r'|salary\s+range'
    r'|compensation\s+range'
    r'|pay\s+range'
    r'|annual\s+base\s+(?:pay|salary)'
    r'|annual\s+salary'
    r'|total\s+(?:annual\s+)?compensation'
    r'|hourly\s+(?:pay\s+)?range'
    r'|pay\s+band'
    r'|typical\s+(?:base\s+)?pay'
    r'|expected\s+(?:base\s+)?(?:pay|salary)'
    r'|target\s+(?:base\s+)?(?:pay|salary)'
    r'|pay\s+(?:is|of|for)'
    r')',
    re.IGNORECASE,
)

# Matches: $161,600  $161.6K  $161k  USD $161,600  USD161,600
_DOLLAR_RE = re.compile(
    r'(?:USD\s*)?\$\s*([\d,]+(?:\.\d+)?)\s*([Kk])?',
    re.IGNORECASE,
)


def _parse_dollars(amount: str, k_suffix: str) -> int:
    try:
        val = float(amount.replace(',', ''))
        return int(val * 1000) if k_suffix else int(val)
    except (ValueError, TypeError):
        return 0


def _extract_salary_from_text(text: str) -> str:
    """
    Scan job description text for salary/pay range mentions.
    Returns a formatted string like "$161,600–$286,200" or "" if nothing found.

    Two strategies:
    1. Look for salary trigger keywords, then grab dollar amounts in a 600-char window.
    2. Fallback: find any two large dollar amounts appearing within 150 chars of each other.
    """
    # Strategy 1: keyword-triggered scan
    for trigger in _SAL_TRIGGER_RE.finditer(text):
        window = text[trigger.start(): trigger.start() + 600]
        matches = _DOLLAR_RE.findall(window)
        amounts = [_parse_dollars(amt, k) for amt, k in matches if _parse_dollars(amt, k) > 0]
        if len(amounts) >= 2:
            lo, hi = amounts[0], amounts[1]
            return f"${lo:,}–${hi:,}" if lo != hi else f"${lo:,}"
        if len(amounts) == 1:
            return f"${amounts[0]:,}"

    # Strategy 2: proximity scan — two dollar amounts > $10k within 150 chars = likely a range
    prev_end = -1
    prev_val = 0
    for m in _DOLLAR_RE.finditer(text):
        val = _parse_dollars(m.group(1), m.group(2))
        if val < 10_000:
            continue
        if prev_val > 0 and (m.start() - prev_end) <= 150:
            lo, hi = min(prev_val, val), max(prev_val, val)
            return f"${lo:,}–${hi:,}" if lo != hi else f"${lo:,}"
        prev_end = m.end()
        prev_val = val

    return ""


# ── HTML description extractor ────────────────────────────────────────────

def _html_desc_to_text(html_str: str) -> str:
    """
    Convert HTML job description to structured plain text.
    Preserves <li> items as '• item', <h*> tags as standalone header lines,
    and <p>/<div> blocks as separate paragraphs.
    """
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "lxml")
    out = []

    def walk(node):
        if not hasattr(node, 'name'):           # NavigableString
            t = str(node).strip()
            if t:
                out.append(t)
            return
        tag = (node.name or "").lower()
        if tag in ("script", "style"):
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            t = node.get_text(separator=" ", strip=True)
            if t:
                out.append("")
                out.append(t)
                out.append("")
        elif tag == "li":
            t = node.get_text(separator=" ", strip=True)
            if t:
                out.append(f"• {t}")
        elif tag in ("ul", "ol"):
            out.append("")
            for child in node.children:
                walk(child)
            out.append("")
        elif tag == "p":
            if node.find(["li", "ul", "ol"]):
                for child in node.children:
                    walk(child)
            else:
                t = node.get_text(separator=" ", strip=True)
                if t:
                    out.append("")
                    out.append(t)
                    out.append("")
        elif tag == "br":
            out.append("")
        else:
            for child in node.children:
                walk(child)

    body = soup.body or soup
    for child in body.children:
        walk(child)

    # Deduplicate consecutive blank lines
    result = []
    prev_blank = True
    for line in out:
        if not line.strip():
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line.strip())
            prev_blank = False

    return "\n".join(result).strip()


# ── Job URL scraper ───────────────────────────────────────────────────────

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Sites known to require JavaScript rendering — scraping will not work
_JS_REQUIRED_SITES = {
    "metacareers.com", "linkedin.com", "indeed.com",
    "careers.google.com", "jobs.lever.co",
    "myworkdayjobs.com", "smartrecruiters.com",
}


def _parse_jsonld(soup):
    """Try to extract JobPosting data from JSON-LD script tags."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or ""
            data = json.loads(raw)
            # Unwrap @graph or list wrappers
            if isinstance(data, dict) and "@graph" in data:
                data = next(
                    (d for d in data["@graph"] if isinstance(d, dict) and d.get("@type") == "JobPosting"),
                    None,
                )
            elif isinstance(data, list):
                data = next(
                    (d for d in data if isinstance(d, dict) and d.get("@type") == "JobPosting"),
                    None,
                )
            if not (data and data.get("@type") == "JobPosting"):
                continue

            result = {"title": "", "company": "", "location": "", "salary": "", "description": ""}

            result["title"] = str(data.get("title", "")).strip()

            org = data.get("hiringOrganization") or {}
            result["company"] = (org.get("name", "") if isinstance(org, dict) else str(org)).strip()

            loc = data.get("jobLocation") or {}
            if isinstance(loc, list):
                loc = loc[0] if loc else {}
            addr = (loc.get("address") or {}) if isinstance(loc, dict) else {}
            if isinstance(addr, str):
                result["location"] = addr
            else:
                parts = [
                    addr.get("addressLocality", ""),
                    addr.get("addressRegion", ""),
                ]
                result["location"] = ", ".join(p for p in parts if p)

            sal = data.get("baseSalary") or data.get("estimatedSalary") or {}
            if isinstance(sal, dict):
                sal_val = sal.get("value") or {}
                cur = "$" if sal.get("currency", "USD") == "USD" else sal.get("currency", "")
                if isinstance(sal_val, (int, float)):
                    result["salary"] = f"{cur}{int(sal_val):,}"
                elif isinstance(sal_val, dict):
                    mn, mx = sal_val.get("minValue"), sal_val.get("maxValue")
                    if mn and mx:
                        result["salary"] = f"{cur}{int(mn):,}–{cur}{int(mx):,}"
                    elif mn:
                        result["salary"] = f"{cur}{int(mn):,}+"

            desc_html = data.get("description", "")
            if desc_html:
                result["description"] = _html_desc_to_text(desc_html)

            # Fallback: mine salary from description text if structured data had none
            if not result["salary"] and result["description"]:
                result["salary"] = _extract_salary_from_text(result["description"])

            if result["title"] or result["description"]:
                return result
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def scrape_job(url: str) -> dict:
    """
    Fetch a job posting URL and extract structured details.
    Returns dict with keys: title, company, location, salary, description, error.
    """
    result = {"title": "", "company": "", "location": "", "salary": "", "description": "", "error": None}

    # Detect known JS-only sites up front
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower().lstrip("www.")
    if any(host == s or host.endswith("." + s) for s in _JS_REQUIRED_SITES):
        result["error"] = (
            f"**{host}** loads job details via JavaScript, so automatic scraping isn't possible. "
            "Please copy the job details from your browser and paste them into the fields below."
        )
        return result

    try:
        resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out. The site took too long to respond."
        return result
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code in (400, 403, 429):
            result["error"] = (
                f"This site blocked the request (HTTP {code}). It likely uses bot protection "
                "or JavaScript rendering. Please paste the job details into the fields below manually."
            )
        elif code == 404:
            result["error"] = "Job posting not found (404). It may have been removed or the URL may be wrong."
        else:
            result["error"] = f"HTTP {code} error. Try opening the URL in your browser and pasting the details manually."
        return result
    except requests.exceptions.RequestException as e:
        result["error"] = f"Could not fetch the URL: {e}"
        return result

    soup = BeautifulSoup(resp.text, "lxml")

    # Layer 1: JSON-LD structured data (most reliable)
    jsonld = _parse_jsonld(soup)
    if jsonld:
        return jsonld

    # Layer 2: Heuristic HTML parsing
    for sel in ["h1.jobTitle", "h1.job-title", "h1[class*='title']",
                ".job-title h1", ".position-title", "h1"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if txt and len(txt) < 200:
                result["title"] = txt
                break

    for sel in ["[class*='company-name']", "[class*='employer']",
                "[class*='organization']", "[class*='company']"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if txt and len(txt) < 100:
                result["company"] = txt
                break

    for sel in ["[class*='location']", "[class*='city']", ".job-location"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if txt and len(txt) < 100:
                result["location"] = txt
                break

    best_desc = ""
    for sel in ["[class*='description']", "[class*='details']",
                "[class*='content']", "article", "main"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(separator="\n").strip()
            if len(txt) > len(best_desc):
                best_desc = txt
    if best_desc:
        # Extract salary from the FULL text before truncating — salary often appears at the end
        result["salary"] = _extract_salary_from_text(best_desc)
        result["description"] = best_desc[:15000]

    if not result["title"] and not result["description"]:
        result["error"] = (
            "Could not extract job details automatically. "
            "This site may block web scraping (e.g. LinkedIn, Indeed). "
            "You can still fill in the fields manually below."
        )
    return result


# ── Word cloud ────────────────────────────────────────────────────────────

_WC_PALETTE = ["#5bc8f5", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#60a5fa"]


def _wc_color(word, font_size, position, orientation, random_state=None, **kwargs):
    idx = abs(hash(word)) % len(_WC_PALETTE)
    return _WC_PALETTE[idx]


def generate_wordcloud(text: str):
    """Generate a word cloud PNG matching the app's dark theme. Returns PNG bytes or None."""
    try:
        wc = WordCloud(
            width=900,
            height=280,
            background_color="#1e2130",
            color_func=_wc_color,
            stopwords=STOP_WORDS,
            max_words=60,
            prefer_horizontal=0.8,
            min_font_size=10,
            collocations=False,
        )
        wc.generate(text)
        fig, ax = plt.subplots(figsize=(9, 2.8))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#1e2130", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


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
                        st.markdown(format_description_md(job["description"]))
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

    if st.session_state.get("job_saved_msg"):
        st.success(st.session_state.job_saved_msg)
        st.session_state.job_saved_msg = ""

    # ── Initialise session state ──────────────────────────────────────────
    if "fetched_job" not in st.session_state:
        st.session_state.fetched_job = {}
    if "fetch_url" not in st.session_state:
        st.session_state.fetch_url = ""

    # ── URL input + fetch button (outside the save form) ─────────────────
    url_col, btn_col = st.columns([5, 1])
    with url_col:
        fetch_url = st.text_input(
            "Job Posting URL",
            value=st.session_state.fetch_url,
            placeholder="Paste the job URL here…",
            label_visibility="collapsed",
        )
    with btn_col:
        fetch_clicked = st.button("🔍 Fetch", use_container_width=True, type="primary")

    if fetch_clicked and fetch_url:
        st.session_state.fetch_url = fetch_url
        with st.spinner("Fetching job details…"):
            fetched = scrape_job(fetch_url)
            st.session_state.fetched_job = fetched
        if fetched.get("error"):
            st.warning(fetched["error"])
        else:
            _FIELD_LABELS = {
                "title": "Job Title", "company": "Company",
                "location": "Location", "salary": "Salary Range",
                "description": "Description",
            }
            filled  = [_FIELD_LABELS[f] for f in _FIELD_LABELS if fetched.get(f)]
            missing = [_FIELD_LABELS[f] for f in _FIELD_LABELS if not fetched.get(f)]
            st.success(f"Fetched: {', '.join(filled)}")
            if missing:
                st.info(
                    f"**Could not auto-fill:** {', '.join(missing)}. "
                    "This site may load these via JavaScript — please enter them manually below."
                )

    fj = st.session_state.fetched_job

    # ── Word cloud ────────────────────────────────────────────────────────
    if fj.get("description"):
        st.markdown("### 🔑 Keyword Cloud")
        wc_bytes = generate_wordcloud(fj["description"])
        if wc_bytes:
            st.image(wc_bytes, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Editable save form ────────────────────────────────────────────────
    with st.form("add_job_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            title    = st.text_input("Job Title *",  value=fj.get("title", ""),    placeholder="e.g. Senior Product Manager")
            company  = st.text_input("Company *",    value=fj.get("company", ""),  placeholder="e.g. Acme Corp")
            location = st.text_input("Location",     value=fj.get("location", ""), placeholder="e.g. Seattle, WA / Remote")
        with c2:
            url      = st.text_input("Job URL",      value=st.session_state.fetch_url, placeholder="https://...")
            salary   = st.text_input("Salary Range", value=fj.get("salary", ""),   placeholder="e.g. $120k–$150k")
            status   = st.selectbox("Status", STATUSES)

        description = st.text_area(
            "Job Description",
            value=fj.get("description", ""),
            placeholder="Auto-filled from URL, or paste manually…",
            height=200,
        )
        notes = st.text_area("Your Notes", placeholder="Why you're interested, referral contact, etc.", height=80)

        submitted = st.form_submit_button("💾 Save Job", use_container_width=True)

    if submitted:
        if not title or not company:
            st.error("Job Title and Company are required.")
        else:
            add_job(title, company, location, url, salary, status, description, notes)
            st.session_state.fetched_job = {}
            st.session_state.fetch_url = ""
            # Store banner in session state, then rerun so Job Board picks up the new row
            st.session_state.job_saved_msg = f"✅ **{title}** at **{company}** saved!"
            st.rerun()


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
