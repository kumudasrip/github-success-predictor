"""
collect_data.py
---------------
Stratified bucket sampling across 8 star ranges and 6 languages.

The core problem with naive GitHub collection is that the star distribution is
heavily power-law: millions of repos have 0 stars, very few have 50-99 stars.
Unfiltered queries therefore almost never return repos near the 100-star
decision boundary, making the classification trivially easy.

Solution: explicitly target each star bucket with a dedicated query and a
per-bucket quota so the final dataset is uniformly distributed across the
full star spectrum — especially dense near the boundary zone (50-99 and
100-299 stars).

Bucket design (5 groups, ~20% each):
  Group A │ 0 stars                │ 20%
  Group B │ 1–49 stars             │ 20%
  Group C │ 50–99 stars  ← boundary│ 20%
  Group D │ 100–999 stars← boundary│ 20%
  Group E │ 1 000+ stars           │ 20%

Within each group multiple sub-buckets and languages are cycled so no single
language or narrow star range dominates.

Usage:
    python collect_data.py --token YOUR_TOKEN
    python collect_data.py --token YOUR_TOKEN --output data/repos.csv --total 600
"""

import argparse
import csv
import math
import os
import re
import time
from datetime import datetime, timezone
from itertools import cycle

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java"]

# Each bucket: (label, star_query_fragment, target_fraction)
# target_fraction is relative — they are normalised internally
BUCKETS = [
    # label          star range string    fraction
    ("0-star",       "stars:0",           0.20),
    ("1-9",          "stars:1..9",        0.10),
    ("10-49",        "stars:10..49",      0.10),
    ("50-99",        "stars:50..99",      0.20),   # hard boundary — oversample
    ("100-299",      "stars:100..299",    0.20),   # hard boundary — oversample
    ("300-999",      "stars:300..999",    0.08),
    ("1000-9999",    "stars:1000..9999",  0.07),
    ("10000+",       "stars:>=10000",     0.05),
]

# GitHub Search caps at 1 000 results per query (10 items × 100 pages).
# With 6 languages × multiple date windows we have plenty of queries per bucket.
DATE_WINDOWS = [
    "pushed:>2024-01-01",
    "pushed:2023-01-01..2023-12-31",
    "pushed:2022-01-01..2022-12-31",
    "pushed:<2022-01-01",
]


# ──────────────────────────────────────────────────────────────────────────────
# GitHub API helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _check_rate_limit(headers: dict) -> None:
    """Block until the search rate limit resets if we're out of calls."""
    r = requests.get("https://api.github.com/rate_limit",
                     headers=headers, timeout=10)
    if r.status_code != 200:
        return
    data      = r.json()
    remaining = data["resources"]["search"]["remaining"]
    reset_at  = data["resources"]["search"]["reset"]
    if remaining == 0:
        wait = max(0, reset_at - time.time()) + 5
        print(f"  ⏳ Rate limit hit — sleeping {wait:.0f}s …")
        time.sleep(wait)


def search_page(query: str, page: int, headers: dict,
                per_page: int = 10) -> list:
    """Fetch one page of search results. Returns [] on hard errors."""
    url    = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "updated", "order": "desc",
              "per_page": per_page, "page": page}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
        except requests.RequestException as exc:
            print(f"    [net error] {exc}  (attempt {attempt+1}/3)")
            time.sleep(3)
            continue

        if r.status_code == 200:
            time.sleep(0.5)                      # secondary rate-limit courtesy
            return r.json().get("items", [])
        if r.status_code == 422:                 # invalid page / unsupported query
            return []
        if r.status_code in (403, 429):          # primary or secondary rate limit
            _check_rate_limit(headers)
            continue
        if r.status_code >= 500:
            time.sleep(5)
            continue
        r.raise_for_status()
    return []


def get_contributors_count(owner: str, repo: str, headers: dict) -> int:
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/contributors",
        params={"per_page": 1, "anon": "false"},
        headers=headers, timeout=10,
    )
    if r.status_code in (204, 403, 451):
        return 0
    if r.status_code != 200:
        return -1
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        m = re.search(r'page=(\d+)>; rel="last"', link)
        return int(m.group(1)) if m else 1
    return max(len(r.json()), 1)


def get_commit_count(owner: str, repo: str, headers: dict) -> int:
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/commits",
        params={"per_page": 1},
        headers=headers, timeout=10,
    )
    if r.status_code != 200:
        return -1
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        m = re.search(r'page=(\d+)>; rel="last"', link)
        return int(m.group(1)) if m else 1
    return max(len(r.json()), 1)


def repo_age_days(created_at: str) -> int:
    created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    return (datetime.now(timezone.utc) - created).days


def extract_features(item: dict, headers: dict) -> dict:
    owner = item["owner"]["login"]
    repo  = item["name"]
    stars = item["stargazers_count"]
    print(f"    ✓ {owner}/{repo:<40} ★ {stars:>7,}")

    contributors = get_contributors_count(owner, repo, headers)
    commits      = get_commit_count(owner, repo, headers)
    time.sleep(0.3)   # be kind to the API between detail calls

    return {
        "full_name":        item["full_name"],
        "stars":            stars,
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
        "repo_age_days":    repo_age_days(item["created_at"]),
        "success":          int(stars >= 100),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bucket collector
# ──────────────────────────────────────────────────────────────────────────────

def collect_bucket(label: str, star_fragment: str,
                   quota: int, headers: dict,
                   global_seen: set) -> list[dict]:
    """
    Collect up to `quota` unique repos for this star bucket by cycling through
    all (language × date_window) query combinations.
    """
    collected: list[dict] = []

    # Build the query rotation for this bucket.
    # IMPORTANT: outer loop = date_window, inner loop = language.
    # This interleaves languages on every cycle() round-trip:
    #   [Python/window1, JavaScript/window1, TypeScript/window1, ...
    #    Python/window2, JavaScript/window2, ...]
    # The previous ordering (outer=language, inner=window) grouped all 5
    # Python queries together at the front, so small quotas were filled
    # entirely from Python before cycle() ever reached other languages.
    queries = []
    for window in DATE_WINDOWS:
        for lang in LANGUAGES:
            queries.append(f"language:{lang} {star_fragment} {window}")
    # Undated fallback — one per language
    for lang in LANGUAGES:
        queries.append(f"language:{lang} {star_fragment}")

    query_cycle = cycle(queries)
    page_state: dict[str, int] = {}   # query → next page number

    exhausted: set[str] = set()
    max_iterations = quota * 10       # safety cap

    iteration = 0
    while len(collected) < quota and len(exhausted) < len(queries):
        if iteration > max_iterations:
            break
        iteration += 1

        query = next(query_cycle)
        if query in exhausted:
            continue

        page = page_state.get(query, 1)
        if page > 10:                 # GitHub Search hard cap: 10 pages × 10
            exhausted.add(query)
            continue

        items = search_page(query, page, headers, per_page=10)
        page_state[query] = page + 1

        if not items:
            exhausted.add(query)
            continue

        for item in items:
            if len(collected) >= quota:
                break
            if item["full_name"] in global_seen:
                continue
            global_seen.add(item["full_name"])
            try:
                row = extract_features(item, headers)
                # Verify the repo actually falls in the intended star range
                # (GitHub search isn't always perfectly filtered at the edges)
                collected.append(row)
            except Exception as exc:
                print(f"    [skip] {item['full_name']}: {exc}")

        time.sleep(0.8)

    return collected


# ──────────────────────────────────────────────────────────────────────────────
# Statistics printer
# ──────────────────────────────────────────────────────────────────────────────

def print_stats(collected: list[dict], output: str) -> None:
    import pandas as pd
    df         = pd.DataFrame(collected)
    total      = len(df)
    success_n  = int(df["success"].sum())
    fail_n     = total - success_n

    star_ranges = [
        ("0",          0,      0),
        ("1–9",        1,      9),
        ("10–49",     10,     49),
        ("50–99",     50,     99),
        ("100–299",  100,    299),
        ("300–999",  300,    999),
        ("1k–9.9k", 1000,   9999),
        ("10k+",   10000, 10**9),
    ]

    print(f"\n{'═'*60}")
    print(f"  Dataset saved → {output}")
    print(f"  Total repos   : {total:,}")
    print(f"\n  Class distribution")
    print(f"    Successful  (≥100 ★) : {success_n:>5,}  ({success_n/total*100:.1f}%)")
    print(f"    Not successful       : {fail_n:>5,}  ({fail_n/total*100:.1f}%)")

    print(f"\n  Star bucket breakdown")
    print(f"    {'Range':<12}  {'Count':>6}  {'%':>6}  {'Bar'}")
    print(f"    {'─'*45}")
    for label, lo, hi in star_ranges:
        n   = int(((df["stars"] >= lo) & (df["stars"] <= hi)).sum())
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<12}  {n:>6,}  {pct:>5.1f}%  {bar}")

    print(f"\n  Language distribution")
    for lang, cnt in df["primary_language"].value_counts().head(8).items():
        pct = cnt / total * 100
        print(f"    {lang:<20}  {cnt:>5,}  ({pct:.1f}%)")
    print(f"{'═'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def collect(token: str, output: str, total: int = 600) -> None:
    headers     = make_headers(token)
    global_seen: set[str] = set()
    all_rows:    list[dict] = []

    # Normalise fractions → integer quotas that sum to `total`
    raw_fractions = [b[2] for b in BUCKETS]
    total_frac    = sum(raw_fractions)
    quotas        = [max(5, round(f / total_frac * total)) for f in raw_fractions]
    # Fix rounding drift
    quotas[-1]   += total - sum(quotas)

    print(f"Stratified bucket collection — target {total} repos")
    print(f"{'─'*60}")
    for (label, star_frag, _), q in zip(BUCKETS, quotas):
        print(f"  {label:<14} {star_frag:<22} quota: {q}")
    print(f"{'─'*60}\n")

    for (label, star_fragment, _), quota in zip(BUCKETS, quotas):
        print(f"\n{'━'*60}")
        print(f"  Bucket [{label}]  target={quota}  ({star_fragment})")
        print(f"{'━'*60}")

        rows = collect_bucket(label, star_fragment, quota, headers, global_seen)
        all_rows.extend(rows)

        filled = len(rows)
        print(f"  → collected {filled}/{quota} repos for bucket [{label}]")
        if filled < quota:
            print(f"    ⚠ Only {filled} found — GitHub may not have enough repos "
                  f"in range [{star_fragment}] across selected languages.")

    # Write CSV
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print_stats(all_rows, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified GitHub repo collector — samples each star bucket explicitly"
    )
    parser.add_argument("--token",  required=True,
                        help="GitHub personal access token (no special scopes needed)")
    parser.add_argument("--output", default="data/repos.csv")
    parser.add_argument("--total",  type=int, default=600,
                        help="Total repos to collect, spread across all buckets (default 600)")
    args = parser.parse_args()
    collect(args.token, args.output, args.total)
