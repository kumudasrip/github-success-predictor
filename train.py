"""
train.py
--------
Cleans data, engineers features, runs EDA (incl. stars histogram, success-rate
by language, class distribution), trains Logistic Regression + Random Forest,
compares performance, and saves the best model.

Usage:
    python train.py --data data/repos.csv
    python train.py --data data/repos.csv --output models/
"""

import argparse
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "forks", "open_issues", "watchers", "size_kb",
    "contributors", "commits", "has_wiki", "has_projects",
    "has_downloads", "topics_count", "repo_age_days",
]
TARGET = "success"

TOP_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
    "C++", "C", "C#", "Ruby", "PHP", "Swift", "Kotlin",
    "Scala", "Shell", "HTML", "CSS", "Jupyter Notebook",
]

# Palette
CLR_FAIL    = "#e07070"
CLR_SUCCESS = "#5b9cf6"
CLR_ACCENT  = "#f5a623"


# ── Data loading & cleaning ───────────────────────────────────────────────────

def load_and_clean(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (cleaned_df_without_stars, stars_series)."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Keep stars aside for EDA before dropping
    stars = df["stars"].copy() if "stars" in df.columns else None

    # Drop leakage columns
    df.drop(columns=["stars", "watchers", "full_name"], errors="ignore", inplace=True)

    # Sentinel → NaN
    df.replace(-1, np.nan, inplace=True)

    # Drop rows without target
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)

    # Cap extreme outliers at 99th percentile
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)

    print(f"After cleaning : {len(df):,} rows")
    vc = df[TARGET].value_counts()
    print(f"Class balance  : {vc[1]} successful ({vc[1]/len(df)*100:.1f}%) | "
          f"{vc[0]} not successful ({vc[0]/len(df)*100:.1f}%)")
    return df, stars


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["commits_per_day"]         = df["commits"] / df["repo_age_days"].replace(0, 1)
    df["fork_to_issue_ratio"]     = df["forks"]   / (df["open_issues"] + 1)
    df["contributors_per_commit"] = df["contributors"] / df["commits"].replace(0, 1)
    df["has_topics"]              = (df["topics_count"] > 0).astype(int)

    df["language_clean"] = df["primary_language"].where(
        df["primary_language"].isin(TOP_LANGUAGES), other="Other"
    )
    lang_dummies = pd.get_dummies(df["language_clean"], prefix="lang")
    df = pd.concat([df, lang_dummies], axis=True)
    df.drop(columns=["primary_language", "language_clean"], inplace=True)
    return df


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_eda(df_raw: pd.DataFrame, stars: pd.Series | None, output_dir: str):
    """
    df_raw : cleaned DataFrame BEFORE feature engineering (still has primary_language)
    stars  : raw star counts (before the column was dropped); may be None
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # ── 1. Class Distribution ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df_raw[TARGET].value_counts().sort_index()
    bars   = ax.bar(
        ["Not Successful\n(< 100 ★)", "Successful\n(≥ 100 ★)"],
        counts.values,
        color=[CLR_FAIL, CLR_SUCCESS], edgecolor="white", linewidth=1.5, width=0.55,
    )
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f"{v:,}\n({v/counts.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution", fontweight="bold", fontsize=13, pad=12)
    ax.set_ylabel("Repository Count")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png", dpi=130)
    plt.close()

    # ── 2. Stars Histogram (log-scale x-axis) ─────────────────────────────────
    if stars is not None and len(stars) > 0:
        stars_clean = stars.dropna().clip(lower=0)
        fig, axes   = plt.subplots(1, 2, figsize=(13, 4.5))

        # Left: raw counts on log x scale
        ax = axes[0]
        bins = np.logspace(0, np.log10(max(stars_clean.max(), 2)), 50)
        ax.hist(stars_clean[stars_clean > 0], bins=bins, color=CLR_ACCENT,
                edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.axvline(100, color="#e05050", linewidth=2, linestyle="--", label="100★ threshold")
        ax.set_xscale("log")
        ax.set_xlabel("Stars (log scale)")
        ax.set_ylabel("Number of Repos")
        ax.set_title("Star Count Distribution (log scale)", fontweight="bold")
        ax.legend()

        # Right: cumulative % to show how many repos fall below 100 stars
        ax2 = axes[1]
        sorted_stars = np.sort(stars_clean)
        cdf = np.arange(1, len(sorted_stars) + 1) / len(sorted_stars) * 100
        ax2.plot(sorted_stars, cdf, color=CLR_ACCENT, linewidth=2)
        ax2.axvline(100, color="#e05050", linewidth=2, linestyle="--", label="100★ threshold")
        pct_below = (stars_clean < 100).sum() / len(stars_clean) * 100
        ax2.axhline(pct_below, color="#aaa", linewidth=1, linestyle=":")
        ax2.set_xscale("log")
        ax2.set_xlabel("Stars (log scale)")
        ax2.set_ylabel("Cumulative % of Repos")
        ax2.set_title("CDF of Star Counts", fontweight="bold")
        ax2.legend()
        ax2.set_ylim(0, 105)

        fig.suptitle("Star Count Distribution", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/stars_histogram.png", dpi=130, bbox_inches="tight")
        plt.close()

    # ── 3. Success Rate by Language ────────────────────────────────────────────
    if "primary_language" in df_raw.columns:
        lang_col  = df_raw["primary_language"].where(
            df_raw["primary_language"].isin(TOP_LANGUAGES), other="Other"
        )
        lang_df   = df_raw[[TARGET]].copy()
        lang_df["language"] = lang_col.values
        lang_stats = (
            lang_df.groupby("language")[TARGET]
            .agg(["mean", "count"])
            .rename(columns={"mean": "success_rate", "count": "n_repos"})
            .query("n_repos >= 5")           # only show languages with enough data
            .sort_values("success_rate", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(9, max(4, len(lang_stats) * 0.45)))
        colors  = [CLR_SUCCESS if r >= 0.5 else CLR_FAIL
                   for r in lang_stats["success_rate"]]
        bars = ax.barh(lang_stats.index, lang_stats["success_rate"] * 100,
                       color=colors, edgecolor="white", linewidth=0.8, height=0.65)
        ax.axvline(50, color="#888", linewidth=1.2, linestyle="--", alpha=0.7)
        for bar, (_, row) in zip(bars, lang_stats.iterrows()):
            ax.text(bar.get_width() + 0.8,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['success_rate']*100:.0f}%  (n={int(row['n_repos'])})",
                    va="center", fontsize=8.5)
        ax.set_xlabel("Success Rate (%)")
        ax.set_title("Success Rate by Primary Language\n(% of repos reaching 100+ stars)",
                     fontweight="bold", fontsize=12)
        ax.set_xlim(0, 110)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rate_by_language.png", dpi=130)
        plt.close()

    # ── 4. Correlation Heatmap ────────────────────────────────────────────────
    num_cols = [c for c in NUMERIC_FEATURES if c in df_raw.columns] + [TARGET]
    corr     = df_raw[num_cols].corr()
    fig, ax  = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.4, square=True,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=130)
    plt.close()

    # ── 5. Feature Distributions by Class ─────────────────────────────────────
    plot_cols = ["forks", "commits", "contributors", "topics_count",
                 "open_issues", "repo_age_days"]
    plot_cols = [c for c in plot_cols if c in df_raw.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes      = axes.flatten()
    for i, col in enumerate(plot_cols[:6]):
        for label, color, name in [
            (0, CLR_FAIL,    "Not Successful"),
            (1, CLR_SUCCESS, "Successful"),
        ]:
            vals = df_raw[df_raw[TARGET] == label][col].dropna()
            axes[i].hist(vals, bins=35, alpha=0.65, color=color,
                         label=name, edgecolor="white", linewidth=0.4)
        axes[i].set_title(col, fontweight="bold")
        axes[i].legend(fontsize=8)
        axes[i].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
    plt.suptitle("Feature Distributions by Class", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=130,
                bbox_inches="tight")
    plt.close()

    print(f"✅  EDA plots saved → {output_dir}/")
    print( "   • class_distribution.png")
    if stars is not None:
        print("   • stars_histogram.png")
    print( "   • success_rate_by_language.png")
    print( "   • correlation_heatmap.png")
    print( "   • feature_distributions.png")


# ── Model building ────────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5, random_state=42
        )),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    return {"Logistic Regression": lr, "Random Forest": rf}


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test,  y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test,    y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test,        y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test,   y_prob), 4),
    }
    print(f"\n── {name} {'─'*(40-len(name))}")
    for k, v in metrics.items():
        if k != "Model":
            bar = "█" * int(v * 20)
            print(f"  {k:<12} {v:.4f}  {bar}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Not Successful", "Successful"]))
    return metrics


def plot_results(results: list, models: dict,
                 X_test, y_test, feature_cols: list, output_dir: str):
    df_res = pd.DataFrame(results)

    # Model comparison bar chart
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x     = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = [CLR_SUCCESS, CLR_ACCENT]
    for i, (_, row) in enumerate(df_res.iterrows()):
        bars = ax.bar(x + i * width, [row[m] for m in metric_names],
                      width, label=row["Model"], color=palette[i], alpha=0.88,
                      edgecolor="white", linewidth=1)
        for bar, v in zip(bars, [row[m] for m in metric_names]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=130)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, model), color in zip(models.items(), [CLR_SUCCESS, CLR_ACCENT]):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}",
                linewidth=2.2, color=color)
    ax.fill_between([0, 1], [0, 1], alpha=0.08, color="grey")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=130)
    plt.close()

    # Feature importance — Random Forest
    rf_model = models.get("Random Forest")
    if rf_model:
        importances = rf_model.named_steps["clf"].feature_importances_
        if len(importances) == len(feature_cols):
            fi = (
                pd.Series(importances, index=feature_cols)
                .sort_values(ascending=False)[:15]
            )
            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.barh(fi.index[::-1], fi.values[::-1],
                           color=CLR_SUCCESS, edgecolor="white",
                           linewidth=0.8, height=0.65)
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 15 Feature Importances  (Random Forest)",
                         fontweight="bold", fontsize=13)
            for bar, v in zip(bars, fi.values[::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=8.5)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=130)
            plt.close()

    print(f"✅  Result plots saved → {output_dir}/")
    print( "   • model_comparison.png")
    print( "   • roc_curves.png")
    print( "   • feature_importance.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    eda_dir = os.path.join(output_dir, "eda_plots")

    # 1. Load & clean  (stars kept separately for EDA)
    df, stars = load_and_clean(data_path)

    # 2. EDA on cleaned data (before feature engineering, still has primary_language)
    run_eda(df, stars, eda_dir)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Prepare X / y
    X            = df.drop(columns=[TARGET])
    y            = df[TARGET]
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 5. Train & evaluate
    pipelines = build_pipelines()
    results   = []
    trained   = {}

    for name, pipe in pipelines.items():
        print(f"\nTraining {name} …")
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        metrics = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)

        cv = cross_val_score(pipe, X, y, cv=5, scoring="f1")
        print(f"  5-Fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")

    # 6. Plots
    plot_results(results, trained, X_test, y_test, feature_cols, output_dir)

    # 7. Save artefacts
    best_name  = max(results, key=lambda r: r["F1 Score"])["Model"]
    best_model = trained[best_name]
    print(f"\n🏆  Best model: {best_name}")

    joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(trained,    os.path.join(output_dir, "all_models.pkl"))
    pd.DataFrame({"feature": feature_cols}).to_csv(
        os.path.join(output_dir, "feature_columns.csv"), index=False
    )
    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "model_results.csv"), index=False
    )

    print(f"\n✅  Saved artefacts → {output_dir}/")
    print(  "    best_model.pkl       ← best-performing model")
    print(  "    all_models.pkl       ← both models")
    print(  "    feature_columns.csv  ← ordered feature list")
    print(  "    model_results.csv    ← performance table")
    print(  "    eda_plots/           ← all EDA & result charts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GitHub Success Predictor")
    parser.add_argument("--data",   default="data/repos.csv")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    main(args.data, args.output)
