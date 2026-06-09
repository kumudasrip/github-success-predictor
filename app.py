"""
app.py  –  GitHub Project Success Predictor
--------------------------------------------
Streamlit web app.

Run:
    streamlit run app.py
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GitHub Success Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0d1117; color: #e6edf3; }

    .hero-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .metric-card h4 {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.3rem;
    }
    .metric-card .value {
        color: #e6edf3;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .prob-bar-bg {
        background: #21262d;
        border-radius: 8px;
        height: 20px;
        overflow: hidden;
        margin: 0.5rem 0 1rem;
    }

    .factor-pos {
        color: #3fb950;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        padding: 2px 0;
    }
    .factor-neg {
        color: #f85149;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        padding: 2px 0;
    }

    .result-success {
        background: linear-gradient(135deg, #0d2a1a 0%, #0d1117 100%);
        border: 1px solid #3fb950;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-fail {
        background: linear-gradient(135deg, #2a0d0d 0%, #0d1117 100%);
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    div[data-testid="stTextInput"] input {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
    }

    .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: 1px solid #2ea043 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #2ea043 !important;
    }

    .divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.5rem 0;
    }

    .tag {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        color: #8b949e;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants (mirror train.py) ───────────────────────────────────────────────

TOP_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
    "C++", "C", "C#", "Ruby", "PHP", "Swift", "Kotlin",
    "Scala", "Shell", "HTML", "CSS", "Jupyter Notebook",
]

MODEL_PATH    = os.path.join("models", "best_model.pkl")
FEATURES_PATH = os.path.join("models", "feature_columns.csv")


# ── GitHub helpers ────────────────────────────────────────────────────────────

def parse_url(url: str):
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s]+)", url.strip())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def make_headers(token):
    h = {"Accept": "application/vnd.github.v3+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h


def get_contributors_count(owner, repo, headers):
    try:
        r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contributors",
            params={"per_page": 1}, headers=headers, timeout=10
        )
        if r.status_code in (204, 403):
            return 0
        if r.status_code != 200:
            return 1
        link = r.headers.get("Link", "")
        if 'rel="last"' in link:
            m = re.search(r'page=(\d+)>; rel="last"', link)
            return int(m.group(1)) if m else 1
        return max(len(r.json()), 1)
    except Exception:
        return 1


def get_commit_count(owner, repo, headers):
    try:
        r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            params={"per_page": 1}, headers=headers, timeout=10
        )
        if r.status_code != 200:
            return 1
        link = r.headers.get("Link", "")
        if 'rel="last"' in link:
            m = re.search(r'page=(\d+)>; rel="last"', link)
            return int(m.group(1)) if m else 1
        return max(len(r.json()), 1)
    except Exception:
        return 1


@st.cache_data(ttl=300, show_spinner=False)
def fetch_repo(owner, repo, token):
    headers = make_headers(token)
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}",
        headers=headers, timeout=15
    )
    if r.status_code == 404:
        return None, "Repository not found."
    if r.status_code == 403:
        return None, "Rate limit exceeded. Add a GitHub token in the sidebar."
    if r.status_code != 200:
        return None, f"GitHub API error {r.status_code}."

    item  = r.json()
    created = datetime.strptime(item["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    age_days     = (datetime.now(timezone.utc) - created).days
    contributors = get_contributors_count(owner, repo, headers)
    commits      = get_commit_count(owner, repo, headers)

    return {
        "forks":            item["forks_count"],
        "open_issues":      item["open_issues_count"],
        "watchers":         item["watchers_count"],
        "size_kb":          item["size"],
        "contributors":     contributors,
        "commits":          commits,
        "has_wiki":         int(item.get("has_wiki", False)),
        "has_projects":     int(item.get("has_projects", False)),
        "has_downloads":    int(item.get("has_downloads", False)),
        "topics_count":     len(item.get("topics", [])),
        "primary_language": item.get("language") or "None",
        "repo_age_days":    age_days,
        # meta
        "_stars":       item["stargazers_count"],
        "_name":        item["full_name"],
        "_description": item.get("description") or "No description.",
        "_language":    item.get("language") or "—",
        "_topics":      item.get("topics", []),
        "_url":         item["html_url"],
        "_avatar":      item["owner"]["avatar_url"],
    }, None


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer(raw, feature_cols):
    df = pd.DataFrame([{k: v for k, v in raw.items() if not k.startswith("_")}])
    df["commits_per_day"]         = df["commits"] / max(df["repo_age_days"].iloc[0], 1)
    df["fork_to_issue_ratio"]     = df["forks"] / (df["open_issues"] + 1)
    df["contributors_per_commit"] = df["contributors"] / max(df["commits"].iloc[0], 1)
    df["has_topics"]              = (df["topics_count"] > 0).astype(int)
    df["language_clean"]          = df["primary_language"].where(
        df["primary_language"].isin(TOP_LANGUAGES), other="Other"
    )
    lang_dummies = pd.get_dummies(df["language_clean"], prefix="lang")
    df = pd.concat([df, lang_dummies], axis=True)
    df.drop(columns=["primary_language", "language_clean"], inplace=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


# ── Explanation ───────────────────────────────────────────────────────────────

def build_explanation(raw):
    pos, neg = [], []
    cpd = raw["commits"] / max(raw["repo_age_days"], 1)

    if raw["contributors"] >= 10:
        pos.append("High contributor activity")
    elif raw["contributors"] <= 1:
        neg.append("Very few contributors")

    if cpd >= 1.0:
        pos.append("Frequent commit activity")
    elif cpd < 0.05:
        neg.append("Very low commit frequency")

    if raw["forks"] >= 20:
        pos.append("Strong fork count")
    elif raw["forks"] <= 2:
        neg.append("Low number of forks")

    if raw["topics_count"] >= 3:
        pos.append("Well-tagged with topics")
    elif raw["topics_count"] == 0:
        neg.append("No topics/tags set")

    if raw["open_issues"] >= 5:
        pos.append("Active issue tracker")

    if raw["repo_age_days"] < 30:
        neg.append("Very new repository")
    elif raw["repo_age_days"] > 365:
        pos.append("Established repository age")

    if raw["size_kb"] >= 1000:
        pos.append("Substantial codebase size")

    return pos, neg


# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    cols  = pd.read_csv(FEATURES_PATH)["feature"].tolist()
    return model, cols


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🚀 GitHub Success Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predict whether a repository will reach 100+ stars using ML</div>', unsafe_allow_html=True)

# Sidebar: token
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    token = st.text_input("GitHub Token (optional)", type="password",
                          help="Increases API rate limit from 60 to 5000 req/hr")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This app uses a trained ML model (Logistic Regression or Random Forest) "
                "to predict GitHub project success based on repository metadata.")
    st.markdown("---")
    st.markdown("**Setup**")
    st.code("python collect_data.py --token TOKEN\npython train.py\nstreamlit run app.py")

# Main input
url_input = st.text_input(
    "Enter a GitHub repository URL",
    placeholder="https://github.com/owner/repository",
    label_visibility="visible"
)

predict_btn = st.button("🔍 Predict Success", use_container_width=True)

model, feature_cols = load_model()

if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ No trained model found. Run `python train.py` first to train the model.")

# ── Prediction flow ───────────────────────────────────────────────────────────

if predict_btn and url_input:
    owner, repo_name = parse_url(url_input)
    if not owner:
        st.error("Invalid GitHub URL. Use: https://github.com/owner/repo")
    elif model is None:
        st.error("Model not loaded. Run `python train.py` first.")
    else:
        with st.spinner("Fetching repository data …"):
            raw, err = fetch_repo(owner, repo_name, token or None)

        if err:
            st.error(f"❌ {err}")
        else:
            X    = engineer(raw, feature_cols)
            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob >= 0.5)
            pos, neg = build_explanation(raw)

            # Result banner
            result_class = "result-success" if pred else "result-fail"
            verdict      = "✅ Likely Successful" if pred else "❌ Unlikely to Succeed"
            verdict_color= "#3fb950" if pred else "#f85149"

            st.markdown(f"""
            <div class="{result_class}">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                    <img src="{raw['_avatar']}" width="40" style="border-radius:50%; border:2px solid #30363d;">
                    <div>
                        <div style="font-weight:700; font-size:1.05rem;">{raw['_name']}</div>
                        <div style="color:#8b949e; font-size:0.85rem;">{raw['_description'][:100]}</div>
                    </div>
                </div>
                <div style="font-size:1.4rem; font-weight:700; color:{verdict_color}; margin-bottom:0.5rem;">
                    {verdict}
                </div>
                <div style="color:#8b949e; font-size:0.85rem; margin-bottom:0.8rem;">
                    Success Probability
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:2.5rem; font-weight:700; color:{verdict_color};">
                    {prob*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar via native streamlit
            st.progress(prob)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⭐ Stars",       f"{raw['_stars']:,}")
            with col2:
                st.metric("🍴 Forks",       f"{raw['forks']:,}")
            with col3:
                st.metric("👥 Contributors", f"{raw['contributors']:,}")
            with col4:
                st.metric("📝 Commits",     f"{raw['commits']:,}")

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("🐛 Open Issues", f"{raw['open_issues']:,}")
            with col6:
                st.metric("🏷️ Topics",      f"{raw['topics_count']}")
            with col7:
                st.metric("📅 Age (days)",   f"{raw['repo_age_days']:,}")
            with col8:
                st.metric("💻 Language",     raw["_language"])

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # Topics
            if raw["_topics"]:
                st.markdown("**Topics**")
                tags_html = " ".join(f'<span class="tag">{t}</span>' for t in raw["_topics"])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("")

            # Explanation
            st.markdown("**Factors**")
            for p in pos:
                st.markdown(f'<div class="factor-pos">+ {p}</div>', unsafe_allow_html=True)
            for n in neg:
                st.markdown(f'<div class="factor-neg">- {n}</div>', unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(f"[🔗 View on GitHub]({raw['_url']})", unsafe_allow_html=False)

# ── Example repos ─────────────────────────────────────────────────────────────

with st.expander("💡 Try these example repositories"):
    examples = [
        "https://github.com/tiangolo/fastapi",
        "https://github.com/streamlit/streamlit",
        "https://github.com/huggingface/transformers",
        "https://github.com/pallets/flask",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.rerun()
