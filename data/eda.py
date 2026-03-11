"""
data/eda.py
-----------
Exploratory Data Analysis module for the SmartRec dataset.

Reads the parquet files produced by data/processing.py and generates:
  - Statistical summaries (returned as dicts)
  - PNG plots saved to reports/figures/
  - JSON summary saved to reports/eda_summary.json

Usage
-----
    python data/eda.py [--processed-dir PATH] [--raw-dir PATH] [--no-save]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent          # data/
PROCESSED_DIR = ROOT / "processed"
RAW_DIR = ROOT / "raw"
REPORTS_DIR = ROOT.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_processed_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three processed parquet files.

    Parameters
    ----------
    processed_dir:
        Directory containing ``interactions.parquet``, ``products.parquet``,
        and ``users.parquet``.

    Returns
    -------
    (interactions, products, users)
    """
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    products = pd.read_parquet(processed_dir / "products.parquet")
    users = pd.read_parquet(processed_dir / "users.parquet")
    logger.info(
        "Loaded %d interactions, %d products, %d users",
        len(interactions),
        len(products),
        len(users),
    )
    return interactions, products, users


def _ensure_figures_dir(figures_dir: Path) -> None:
    """Create the figures output directory if it does not exist."""
    figures_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# numpy JSON serialiser
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_rating_distribution(
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> dict[str, float]:
    """Analyse and plot the distribution of ratings.

    Parameters
    ----------
    interactions:
        Interactions DataFrame with a ``rating`` column.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``rating_distribution.png`` to *figures_dir*.

    Returns
    -------
    dict with keys: mean, median, std, min, max, pct_1 … pct_5
    """
    import matplotlib.pyplot as plt

    ratings = interactions["rating"].dropna()

    counts = ratings.value_counts().sort_index()
    stats: dict[str, float] = {
        "mean": float(ratings.mean()),
        "median": float(ratings.median()),
        "std": float(ratings.std()),
        "min": float(ratings.min()),
        "max": float(ratings.max()),
    }
    for val, cnt in counts.items():
        stats[f"pct_{int(val)}"] = float(cnt / len(ratings))

    if save:
        _ensure_figures_dir(figures_dir)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart of counts
        axes[0].bar(counts.index, counts.values, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("Rating")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Rating Count by Value")
        axes[0].set_xticks(counts.index)

        # Percentage pie chart
        axes[1].pie(
            counts.values,
            labels=[f"{r:.0f}★" for r in counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title("Rating Distribution (%)")

        fig.suptitle("Rating Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = figures_dir / "rating_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_user_activity(
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> dict[str, float]:
    """Analyse and plot the distribution of reviews per user.

    Parameters
    ----------
    interactions:
        Interactions DataFrame.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``user_activity_distribution.png`` to *figures_dir*.

    Returns
    -------
    dict with keys: mean, median, std, min, max, p25, p75, p95, p99
    """
    import matplotlib.pyplot as plt

    activity = interactions.groupby("user_id")["rating"].count()

    stats: dict[str, float] = {
        "mean": float(activity.mean()),
        "median": float(activity.median()),
        "std": float(activity.std()),
        "min": float(activity.min()),
        "max": float(activity.max()),
        "p25": float(activity.quantile(0.25)),
        "p75": float(activity.quantile(0.75)),
        "p95": float(activity.quantile(0.95)),
        "p99": float(activity.quantile(0.99)),
    }

    if save:
        _ensure_figures_dir(figures_dir)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Histogram (log scale x)
        axes[0].hist(activity, bins=50, color="darkorange", edgecolor="white", log=True)
        axes[0].set_xlabel("Reviews per User")
        axes[0].set_ylabel("Number of Users (log)")
        axes[0].set_title("User Review Count Histogram")

        # Box plot
        axes[1].boxplot(activity, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="darkorange", alpha=0.6))
        axes[1].set_ylabel("Reviews per User")
        axes[1].set_title("User Activity Box Plot")

        fig.suptitle("User Activity Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = figures_dir / "user_activity_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_product_activity(
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> dict[str, float]:
    """Analyse and plot the distribution of reviews per product.

    Parameters
    ----------
    interactions:
        Interactions DataFrame.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``product_activity_distribution.png`` to *figures_dir*.

    Returns
    -------
    dict with keys: mean, median, std, min, max, p25, p75, p95, p99
    """
    import matplotlib.pyplot as plt

    activity = interactions.groupby("product_id")["rating"].count()

    stats: dict[str, float] = {
        "mean": float(activity.mean()),
        "median": float(activity.median()),
        "std": float(activity.std()),
        "min": float(activity.min()),
        "max": float(activity.max()),
        "p25": float(activity.quantile(0.25)),
        "p75": float(activity.quantile(0.75)),
        "p95": float(activity.quantile(0.95)),
        "p99": float(activity.quantile(0.99)),
    }

    if save:
        _ensure_figures_dir(figures_dir)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].hist(activity, bins=50, color="seagreen", edgecolor="white", log=True)
        axes[0].set_xlabel("Reviews per Product")
        axes[0].set_ylabel("Number of Products (log)")
        axes[0].set_title("Product Review Count Histogram")

        axes[1].boxplot(activity, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="seagreen", alpha=0.6))
        axes[1].set_ylabel("Reviews per Product")
        axes[1].set_title("Product Activity Box Plot")

        fig.suptitle("Product Activity Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = figures_dir / "product_activity_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_temporal_trends(
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> dict[str, object]:
    """Analyse and plot monthly interaction and average rating trends over time.

    Skipped gracefully if the ``timestamp`` column is absent or entirely null.

    Parameters
    ----------
    interactions:
        Interactions DataFrame; expects an optional ``timestamp`` Unix column.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``temporal_trends.png`` to *figures_dir*.

    Returns
    -------
    dict with keys: min_date, max_date, n_months, peak_month, peak_count
        or ``{"skipped": True}`` if timestamp data is unavailable.
    """
    import matplotlib.pyplot as plt

    if "timestamp" not in interactions.columns or interactions["timestamp"].isna().all():
        logger.warning("analyze_temporal_trends: no timestamp data — skipping")
        return {"skipped": True}

    ts = pd.to_datetime(interactions["timestamp"].dropna(), unit="s", errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        logger.warning("analyze_temporal_trends: timestamp column unparseable — skipping")
        return {"skipped": True}

    df_t = interactions.loc[ts.index].copy()
    df_t["date"] = ts.dt.to_period("M")

    monthly = df_t.groupby("date").agg(
        count=("rating", "count"),
        avg_rating=("rating", "mean"),
    )
    peak_idx = monthly["count"].idxmax()

    stats: dict[str, object] = {
        "min_date": str(monthly.index.min()),
        "max_date": str(monthly.index.max()),
        "n_months": int(len(monthly)),
        "peak_month": str(peak_idx),
        "peak_count": int(monthly["count"].max()),
    }

    if save:
        _ensure_figures_dir(figures_dir)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        x = monthly.index.astype(str)
        axes[0].bar(x, monthly["count"], color="royalblue", alpha=0.8)
        axes[0].set_ylabel("Interactions")
        axes[0].set_title("Monthly Interactions")

        axes[1].plot(x, monthly["avg_rating"], color="crimson", marker="o", markersize=3)
        axes[1].set_ylabel("Average Rating")
        axes[1].set_xlabel("Month")
        axes[1].set_title("Monthly Average Rating")
        axes[1].set_ylim(1, 5)

        # Reduce x-tick density for readability
        tick_step = max(1, len(x) // 20)
        axes[1].set_xticks(range(0, len(x), tick_step))
        axes[1].set_xticklabels(x[::tick_step], rotation=45, ha="right")

        fig.suptitle("Temporal Trends", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = figures_dir / "temporal_trends.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_sparsity(
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
    sample_size: int = 100,
) -> dict[str, float]:
    """Compute sparsity metrics and plot a user–item heatmap sample.

    The heatmap shows ``sample_size`` × ``sample_size`` most-active
    users and products to avoid memory issues with large matrices.

    Parameters
    ----------
    interactions:
        Interactions DataFrame.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``sparsity_heatmap.png`` to *figures_dir*.
    sample_size:
        Number of top users/products to include in the heatmap.

    Returns
    -------
    dict with keys: n_users, n_products, n_interactions, sparsity,
        density, sample_sparsity
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_users = interactions["user_id"].nunique()
    n_products = interactions["product_id"].nunique()
    n_interactions = len(interactions)
    possible = n_users * n_products
    density = n_interactions / possible if possible > 0 else 0.0
    spar = 1.0 - density

    stats: dict[str, float] = {
        "n_users": float(n_users),
        "n_products": float(n_products),
        "n_interactions": float(n_interactions),
        "sparsity": spar,
        "density": density,
    }

    if save:
        _ensure_figures_dir(figures_dir)

        # Build sample matrix from most-active users and products
        top_users = (
            interactions.groupby("user_id")["rating"]
            .count()
            .nlargest(sample_size)
            .index
        )
        top_products = (
            interactions.groupby("product_id")["rating"]
            .count()
            .nlargest(sample_size)
            .index
        )
        sample = interactions[
            interactions["user_id"].isin(top_users)
            & interactions["product_id"].isin(top_products)
        ]
        matrix = (
            sample.pivot_table(
                index="user_id",
                columns="product_id",
                values="rating",
                aggfunc="mean",
            )
            .reindex(index=top_users, columns=top_products)
        )

        sample_density = matrix.notna().values.mean()
        stats["sample_sparsity"] = float(1.0 - sample_density)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            matrix.notna().astype(int),
            ax=ax,
            cmap="Blues",
            cbar_kws={"label": "Has interaction"},
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_xlabel(f"Top-{sample_size} Products")
        ax.set_ylabel(f"Top-{sample_size} Users")
        ax.set_title(
            f"User–Item Interaction Heatmap (sample {sample_size}×{sample_size})\n"
            f"Global sparsity: {spar:.2%}  |  Sample density: {sample_density:.2%}"
        )
        fig.tight_layout()
        out = figures_dir / "sparsity_heatmap.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_filter_impact(
    raw_dir: Path,
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> dict[str, object]:
    """Compare raw vs. processed record counts to show filtering impact.

    Parameters
    ----------
    raw_dir:
        Directory containing ``reviews.csv`` (raw data).
    interactions:
        Final processed interactions DataFrame.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``filter_impact.png`` to *figures_dir*.

    Returns
    -------
    dict with keys: n_raw, n_processed, pct_retained, n_dropped
    """
    import matplotlib.pyplot as plt

    raw_path = raw_dir / "reviews.csv"
    if not raw_path.exists():
        logger.warning("analyze_filter_impact: %s not found — using 0 as raw count", raw_path)
        n_raw = 0
    else:
        # Count rows without loading full file into memory
        with open(raw_path, "rb") as f:
            n_raw = sum(1 for _ in f) - 1  # subtract header

    n_processed = len(interactions)
    n_dropped = max(0, n_raw - n_processed)
    pct_retained = (n_processed / n_raw * 100) if n_raw > 0 else 0.0

    stats: dict[str, object] = {
        "n_raw": int(n_raw),
        "n_processed": int(n_processed),
        "pct_retained": float(pct_retained),
        "n_dropped": int(n_dropped),
    }

    if save:
        _ensure_figures_dir(figures_dir)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = ["Raw", "Processed"]
        values = [n_raw, n_processed]
        colors = ["#e74c3c", "#2ecc71"]

        axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
        axes[0].set_ylabel("Record Count")
        axes[0].set_title("Raw vs Processed Records")
        for i, v in enumerate(values):
            axes[0].text(i, v * 1.01, f"{v:,}", ha="center", fontsize=10)

        # Donut chart: retained vs dropped
        retained = n_processed
        dropped = n_dropped
        wedge_sizes = [retained, dropped] if dropped > 0 else [retained, 0]
        wedge_labels = [
            f"Retained\n{pct_retained:.1f}%",
            f"Dropped\n{100 - pct_retained:.1f}%",
        ]
        wedge_colors = ["#2ecc71", "#e74c3c"]
        axes[1].pie(
            wedge_sizes,
            labels=wedge_labels,
            colors=wedge_colors,
            startangle=90,
            wedgeprops=dict(width=0.5),
        )
        axes[1].set_title("Filter Impact (Donut)")

        fig.suptitle("Filter Impact: Raw → Processed", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = figures_dir / "filter_impact.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_categories(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
    top_n: int = 15,
) -> dict[str, object]:
    """Analyse category distribution weighted by interaction count.

    Skipped gracefully if ``products`` has no ``category`` column or all
    values are null.

    Parameters
    ----------
    products:
        Products DataFrame with an optional ``category`` column.
    interactions:
        Interactions DataFrame (used to weight categories by activity).
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``category_distribution.png`` to *figures_dir*.
    top_n:
        Number of top categories to display.

    Returns
    -------
    dict with keys: n_categories, top_categories (list of {category, n_products, n_interactions})
        or ``{"skipped": True}`` if category data is unavailable.
    """
    import matplotlib.pyplot as plt

    if "category" not in products.columns or products["category"].isna().all():
        logger.warning("analyze_categories: no category data — skipping")
        return {"skipped": True}

    cat_col = products[["product_id", "category"]].dropna(subset=["category"])
    # Merge with interaction counts
    interaction_counts = (
        interactions.groupby("product_id")["rating"]
        .count()
        .rename("n_interactions")
        .reset_index()
    )
    merged = cat_col.merge(interaction_counts, on="product_id", how="left")
    merged["n_interactions"] = merged["n_interactions"].fillna(0)

    cat_stats = (
        merged.groupby("category")
        .agg(n_products=("product_id", "count"), n_interactions=("n_interactions", "sum"))
        .sort_values("n_interactions", ascending=False)
    )
    top = cat_stats.head(top_n)

    stats: dict[str, object] = {
        "n_categories": int(len(cat_stats)),
        "top_categories": [
            {
                "category": cat,
                "n_products": int(row["n_products"]),
                "n_interactions": int(row["n_interactions"]),
            }
            for cat, row in top.iterrows()
        ],
    }

    if save:
        _ensure_figures_dir(figures_dir)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Horizontal bar: interactions
        axes[0].barh(top.index[::-1], top["n_interactions"][::-1], color="mediumpurple")
        axes[0].set_xlabel("Number of Interactions")
        axes[0].set_title(f"Top {top_n} Categories by Interactions")

        # Horizontal bar: products
        axes[1].barh(top.index[::-1], top["n_products"][::-1], color="orchid")
        axes[1].set_xlabel("Number of Products")
        axes[1].set_title(f"Top {top_n} Categories by Products")

        fig.suptitle("Category Distribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out = figures_dir / "category_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return stats


def analyze_rating_activity_correlation(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> dict[str, float]:
    """Analyse the correlation between user activity and average rating.

    Uses a hexbin plot to handle large datasets without overplotting.

    Parameters
    ----------
    interactions:
        Interactions DataFrame.
    users:
        Users DataFrame with ``total_reviews`` and ``avg_rating`` columns.
    figures_dir:
        Directory where the PNG is saved.
    save:
        If ``True``, write ``rating_activity_correlation.png`` to *figures_dir*.

    Returns
    -------
    dict with keys: pearson_r, spearman_r
    """
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    df = users[["total_reviews", "avg_rating"]].dropna()

    pearson_r, _ = sp_stats.pearsonr(df["total_reviews"], df["avg_rating"])
    spearman_r, _ = sp_stats.spearmanr(df["total_reviews"], df["avg_rating"])

    result: dict[str, float] = {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
    }

    if save:
        _ensure_figures_dir(figures_dir)
        fig, ax = plt.subplots(figsize=(9, 7))

        hb = ax.hexbin(
            df["total_reviews"],
            df["avg_rating"],
            gridsize=40,
            cmap="YlOrRd",
            mincnt=1,
        )
        fig.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel("Total Reviews per User")
        ax.set_ylabel("Average Rating")
        ax.set_title(
            f"User Activity vs Average Rating\n"
            f"Pearson r = {pearson_r:.3f}  |  Spearman ρ = {spearman_r:.3f}"
        )
        fig.tight_layout()
        out = figures_dir / "rating_activity_correlation.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", out)

    return result


# ---------------------------------------------------------------------------
# Summary export
# ---------------------------------------------------------------------------

def save_eda_summary(
    summary: dict[str, Any],
    output_path: Path | None = None,
) -> None:
    """Serialise the EDA summary dict to a JSON file.

    Handles ``numpy.integer`` / ``numpy.floating`` types via a custom encoder.

    Parameters
    ----------
    summary:
        Dictionary of EDA results (nested dicts returned by ``run``).
    output_path:
        Destination path. Defaults to ``reports/eda_summary.json``.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "eda_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    logger.info("EDA summary saved to %s", output_path)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(
    processed_dir: Path = PROCESSED_DIR,
    raw_dir: Path = RAW_DIR,
    save_figures: bool = True,
) -> dict[str, Any]:
    """Run the full EDA pipeline.

    Parameters
    ----------
    processed_dir:
        Directory with the parquet files from ``data/processing.py``.
    raw_dir:
        Directory with the original ``reviews.csv`` (used by
        ``analyze_filter_impact``).
    save_figures:
        Whether to save PNG plots to ``reports/figures/``.

    Returns
    -------
    dict
        Complete summary with one key per analysis function.
    """
    interactions, products, users = load_processed_data(processed_dir)

    summary: dict[str, Any] = {}

    logger.info("Analyzing rating distribution …")
    summary["rating_distribution"] = analyze_rating_distribution(
        interactions, save=save_figures
    )

    logger.info("Analyzing user activity …")
    summary["user_activity"] = analyze_user_activity(
        interactions, save=save_figures
    )

    logger.info("Analyzing product activity …")
    summary["product_activity"] = analyze_product_activity(
        interactions, save=save_figures
    )

    logger.info("Analyzing temporal trends …")
    summary["temporal_trends"] = analyze_temporal_trends(
        interactions, save=save_figures
    )

    logger.info("Analyzing sparsity …")
    summary["sparsity"] = analyze_sparsity(interactions, save=save_figures)

    logger.info("Analyzing filter impact …")
    summary["filter_impact"] = analyze_filter_impact(
        raw_dir, interactions, save=save_figures
    )

    logger.info("Analyzing categories …")
    summary["categories"] = analyze_categories(
        products, interactions, save=save_figures
    )

    logger.info("Analyzing rating–activity correlation …")
    summary["rating_activity_correlation"] = analyze_rating_activity_correlation(
        interactions, users, save=save_figures
    )

    save_eda_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SmartRec EDA")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Path to processed parquet directory (default: data/processed/)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Path to raw data directory (default: data/raw/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving PNG figures",
    )
    args = parser.parse_args()

    result = run(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        save_figures=not args.no_save,
    )
    print(json.dumps(result, indent=2, cls=_NumpyEncoder))
