"""
predict.py
----------
Fetches live metadata for a GitHub repo and runs the trained model.

Usage:
    python predict.py --url https://github.com/tiangolo/fastapi
    python predict.py --url https://github.com/tiangolo/fastapi --token YOUR_TOKEN
"""

import argparse
import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = os.path.join("models", "best_model.pkl")
FEATURES_PATH = os.path.join("models", "feature_columns.csv")

TOP_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
    "C++", "C", "C#", "Ruby", "PHP", "Swift", "Kotlin",
    "Scala", "Shell", "HTML", "CSS", "Jupyter Notebook",
]


# ── GitHub fetching ───────────────────────────────────────────────────────────

def parse_repo_url(url: str) -> tuple[str, str]:
    url = url.rstrip("/")
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return m.group(1), m.group(2)


def get_headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h


def get_contributors_count(owner, repo, headers) -> int:
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    r = requests.get(url, params={"per_page": 1}, headers=headers, timeout=10)
    if r.status_code in (204, 403):
        return 0
    if r.status_code != 200:
        return 1
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        m = re.search(r'page=(\d+)>; rel="last"', link)
        return int(m.group(1)) if m else 1
    return max(len(r.json()), 1)


def get_commit_count(owner, repo, headers) -> int:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    r = requests.get(url, params={"per_page": 1}, headers=headers, timeout=10)
    if r.status_code != 200:
        return 1
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        m = re.search(r'page=(\d+)>; rel="last"', link)
        return int(m.group(1)) if m else 1
    return max(len(r.json()), 1)


def fetch_repo_features(owner: str, repo: str, token: str | None) -> dict:
    headers = get_headers(token)
    url     = f"https://api.github.com/repos/{owner}/{repo}"

    print(f"Fetching {owner}/{repo} …")
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 404:
        raise ValueError(f"Repository not found: {owner}/{repo}")
    r.raise_for_status()
    item = r.json()

    created = datetime.strptime(item["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    age_days     = (datetime.now(timezone.utc) - created).days
    contributors = get_contributors_count(owner, repo, headers)
    commits      = get_commit_count(owner, repo, headers)

    raw = {
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
        # meta (not fed to model)
        "_stars":           item["stargazers_count"],
        "_name":            item["full_name"],
        "_description":     item.get("description") or "",
        "_language":        item.get("language") or "None",
        "_topics":          item.get("topics", []),
    }
    return raw


# ── Feature engineering (must mirror train.py) ────────────────────────────────

def engineer_features(raw: dict, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in raw.items() if not k.startswith("_")}])

    df["commits_per_day"]         = df["commits"] / (df["repo_age_days"].replace(0, 1))
    df["fork_to_issue_ratio"]     = df["forks"] / (df["open_issues"] + 1)
    df["contributors_per_commit"] = df["contributors"] / (df["commits"].replace(0, 1))
    df["has_topics"]              = (df["topics_count"] > 0).astype(int)

    df["language_clean"] = df["primary_language"].where(
        df["primary_language"].isin(TOP_LANGUAGES), other="Other"
    )
    lang_dummies = pd.get_dummies(df["language_clean"], prefix="lang")
    df = pd.concat([df, lang_dummies], axis=True)
    df.drop(columns=["primary_language", "language_clean"], inplace=True)

    # Align to training columns (add missing, drop extra)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df


# ── Explanation helpers ───────────────────────────────────────────────────────

def explain(raw: dict, prob: float) -> list[str]:
    lines = []
    forks        = raw["forks"]
    contributors = raw["contributors"]
    commits      = raw["commits"]
    age          = raw["repo_age_days"]
    topics       = raw["topics_count"]
    open_issues  = raw["open_issues"]

    commits_per_day = commits / max(age, 1)

    if contributors >= 10:
        lines.append("+ High contributor activity")
    elif contributors <= 1:
        lines.append("- Very few contributors")

    if commits_per_day >= 1.0:
        lines.append("+ Frequent commit activity")
    elif commits_per_day < 0.05:
        lines.append("- Very low commit frequency")

    if forks >= 20:
        lines.append("+ Strong fork count (community interest)")
    elif forks <= 2:
        lines.append("- Low number of forks")

    if topics >= 3:
        lines.append("+ Well-tagged with topics")
    elif topics == 0:
        lines.append("- No topics/tags set")

    if open_issues >= 5:
        lines.append("+ Active issue tracker (engagement)")

    if age < 30:
        lines.append("- Very new repository")
    elif age > 365:
        lines.append("+ Established repository age")

    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(url: str, token: str | None = None):
    # Load model & feature list
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    model         = joblib.load(MODEL_PATH)
    feature_cols  = pd.read_csv(FEATURES_PATH)["feature"].tolist()

    owner, repo = parse_repo_url(url)
    raw         = fetch_repo_features(owner, repo, token)
    X           = engineer_features(raw, feature_cols)

    prob        = model.predict_proba(X)[0, 1]
    pred        = int(prob >= 0.5)
    explanation = explain(raw, prob)

    # Print result
    print("\n" + "═" * 50)
    print(f"  Repository : {raw['_name']}")
    print(f"  Description: {raw['_description'][:80]}")
    print(f"  Language   : {raw['_language']}")
    if raw["_topics"]:
        print(f"  Topics     : {', '.join(raw['_topics'][:5])}")
    print("─" * 50)
    print(f"  Success Probability : {prob*100:.1f}%")
    print(f"  Predicted Class     : {'✅ Successful' if pred else '❌ Not Successful'}")
    print("─" * 50)
    print("  Factors:")
    for line in explanation:
        print(f"    {line}")
    print("═" * 50 + "\n")

    return {
        "repo":        raw["_name"],
        "probability": round(prob, 4),
        "prediction":  pred,
        "explanation": explanation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict GitHub repo success")
    parser.add_argument("--url",   required=True, help="GitHub repo URL")
    parser.add_argument("--token", default=None,   help="GitHub personal access token")
    args = parser.parse_args()
    predict(args.url, args.token)
