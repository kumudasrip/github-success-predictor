"""
generate_demo_data.py
---------------------
Generates a REALISTIC synthetic dataset using the same stratified bucket design
as collect_data.py so offline testing produces representative ML difficulty.

Bucket design (mirrors collect_data.py):
  0 stars       20%   → feature profile: very_low
  1–49 stars    20%   → feature profile: low          (two sub-buckets)
  50–99 stars   20%   → feature profile: boundary     ← overlapping zone
  100–299 stars 20%   → feature profile: boundary     ← overlapping zone
  300–9999      13%   → feature profile: high
  10000+         7%   → feature profile: very_high

Features within each profile are drawn from overlapping distributions so
the 50–99 / 100–299 boundary zone is genuinely hard to classify.

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --output data/repos.csv --n 600
"""

import argparse
import os
import numpy as np
import pandas as pd

LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust",
    "Java", "C++", "C", "Ruby", "PHP", "Swift", "Kotlin", "None",
]

# (label, star_lo, star_hi, fraction, feature_profile)
BUCKETS = [
    ("0-star",    0,       0,     0.20, "very_low"),
    ("1-9",       1,       9,     0.10, "low"),
    ("10-49",    10,      49,     0.10, "low"),
    ("50-99",    50,      99,     0.20, "boundary"),   # hard zone
    ("100-299", 100,     299,     0.20, "boundary"),   # hard zone
    ("300-999", 300,     999,     0.08, "high"),
    ("1k-9.9k", 1000,   9999,    0.07, "high"),
    ("10k+",   10000, 100000,    0.05, "very_high"),
]

SEED = 42
rng  = np.random.default_rng(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Feature samplers — distributions deliberately overlap in the boundary zone
# ──────────────────────────────────────────────────────────────────────────────

def _rint(lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))

def _lognorm(mu: float, sigma: float, lo: int = 0, hi: int | None = None) -> int:
    v = int(np.exp(rng.normal(mu, sigma)))
    v = max(v, lo)
    return min(v, hi) if hi is not None else v

def _poisson(lam: float) -> int:
    return max(0, int(rng.poisson(lam)))


PROFILES: dict[str, dict] = {
    "very_low": dict(
        contributors  = lambda: _poisson(0.5),
        commits       = lambda: _lognorm(1.5, 0.8, lo=1, hi=100),
        forks         = lambda: _poisson(0.2),
        open_issues   = lambda: _poisson(0.5),
        topics_count  = lambda: _rint(0, 1),
        repo_age_days = lambda: _rint(1, 600),
        size_kb       = lambda: _lognorm(2.5, 1.2, lo=1, hi=2000),
        has_wiki      = lambda: int(rng.random() < 0.08),
        has_projects  = lambda: int(rng.random() < 0.05),
    ),
    "low": dict(
        contributors  = lambda: _poisson(1.2),
        commits       = lambda: _lognorm(2.8, 0.9, lo=1, hi=500),
        forks         = lambda: _lognorm(1.0, 0.8, lo=0, hi=50),
        open_issues   = lambda: _poisson(2.0),
        topics_count  = lambda: _rint(0, 3),
        repo_age_days = lambda: _rint(10, 1000),
        size_kb       = lambda: _lognorm(4.0, 1.3, lo=1, hi=8000),
        has_wiki      = lambda: int(rng.random() < 0.20),
        has_projects  = lambda: int(rng.random() < 0.12),
    ),
    # boundary zone: distributions deliberately straddle both classes
    "boundary": dict(
        contributors  = lambda: _lognorm(1.5, 0.9, lo=1, hi=150),
        commits       = lambda: _lognorm(3.8, 1.1, lo=5, hi=3000),
        forks         = lambda: _lognorm(2.5, 1.2, lo=0, hi=500),
        open_issues   = lambda: _lognorm(2.8, 1.1, lo=0, hi=300),
        topics_count  = lambda: _rint(0, 6),
        repo_age_days = lambda: _rint(30, 1500),
        size_kb       = lambda: _lognorm(5.5, 1.5, lo=10, hi=40000),
        has_wiki      = lambda: int(rng.random() < 0.40),
        has_projects  = lambda: int(rng.random() < 0.25),
    ),
    "high": dict(
        contributors  = lambda: _lognorm(2.8, 1.0, lo=3, hi=500),
        commits       = lambda: _lognorm(5.2, 1.1, lo=30, hi=20000),
        forks         = lambda: _lognorm(4.2, 1.2, lo=10, hi=10000),
        open_issues   = lambda: _lognorm(3.5, 1.1, lo=2, hi=2000),
        topics_count  = lambda: _rint(2, 9),
        repo_age_days = lambda: _rint(90, 2500),
        size_kb       = lambda: _lognorm(7.0, 1.4, lo=200, hi=100000),
        has_wiki      = lambda: int(rng.random() < 0.65),
        has_projects  = lambda: int(rng.random() < 0.45),
    ),
    "very_high": dict(
        contributors  = lambda: _lognorm(4.0, 1.0, lo=20, hi=5000),
        commits       = lambda: _lognorm(7.0, 1.2, lo=500, hi=100000),
        forks         = lambda: _lognorm(6.0, 1.0, lo=500, hi=200000),
        open_issues   = lambda: _lognorm(5.0, 1.1, lo=50, hi=20000),
        topics_count  = lambda: _rint(3, 10),
        repo_age_days = lambda: _rint(365, 4000),
        size_kb       = lambda: _lognorm(9.0, 1.3, lo=1000, hi=500000),
        has_wiki      = lambda: int(rng.random() < 0.90),
        has_projects  = lambda: int(rng.random() < 0.75),
    ),
}


def sample_row(profile_name: str, stars: int, idx: int) -> dict:
    p = PROFILES[profile_name]
    return {
        "full_name":        f"synthetic/repo_{idx}",
        "stars":            stars,
        "forks":            p["forks"](),
        "open_issues":      p["open_issues"](),
        "watchers":         max(0, stars + _rint(-5, 5)),
        "size_kb":          p["size_kb"](),
        "contributors":     p["contributors"](),
        "commits":          p["commits"](),
        "has_wiki":         p["has_wiki"](),
        "has_projects":     p["has_projects"](),
        "has_downloads":    int(rng.random() < 0.35),
        "topics_count":     p["topics_count"](),
        "primary_language": str(rng.choice(LANGUAGES)),
        "repo_age_days":    p["repo_age_days"](),
        "success":          int(stars >= 100),
    }


def generate(n: int = 600) -> pd.DataFrame:
    # Compute per-bucket counts from fractions
    fractions = [b[3] for b in BUCKETS]
    total_f   = sum(fractions)
    counts    = [max(3, round(f / total_f * n)) for f in fractions]
    counts[-1] += n - sum(counts)   # absorb rounding error in last bucket

    records = []
    for (label, lo, hi, _, profile), count in zip(BUCKETS, counts):
        for _ in range(count):
            stars = _rint(lo, hi)
            records.append(sample_row(profile, stars, len(records)))

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def print_stats(df: pd.DataFrame) -> None:
    total     = len(df)
    success_n = int(df["success"].sum())
    fail_n    = total - success_n

    star_ranges = [
        ("0",        0,     0),
        ("1–9",      1,     9),
        ("10–49",   10,    49),
        ("50–99",   50,    99),
        ("100–299", 100,  299),
        ("300–999", 300,  999),
        ("1k–9.9k", 1000, 9999),
        ("10k+",   10000, 10**9),
    ]

    print(f"\n{'═'*58}")
    print(f"  Total rows       : {total:,}")
    print(f"\n  Class distribution")
    print(f"    Successful  (≥100★) : {success_n:>5,}  ({success_n/total*100:.1f}%)")
    print(f"    Not successful      : {fail_n:>5,}  ({fail_n/total*100:.1f}%)")
    print(f"\n  Star bucket breakdown")
    print(f"    {'Range':<12}  {'Count':>6}  {'%':>6}  {'Bar'}")
    print(f"    {'─'*43}")
    for label, lo, hi in star_ranges:
        n   = int(((df["stars"] >= lo) & (df["stars"] <= hi)).sum())
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<12}  {n:>6,}  {pct:>5.1f}%  {bar}")
    print(f"\n  Language distribution (top 6)")
    for lang, cnt in df["primary_language"].value_counts().head(6).items():
        print(f"    {lang:<20}  {cnt:>5,}  ({cnt/total*100:.1f}%)")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/repos.csv")
    parser.add_argument("--n",      type=int, default=600)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = generate(args.n)
    df.to_csv(args.output, index=False)
    print(f"✅  Generated {len(df)} rows → {args.output}")
    print_stats(df)
