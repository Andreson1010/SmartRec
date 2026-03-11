"""
data/processing.py
------------------
Preprocessing pipeline for the Amazon Electronics Reviews dataset.

Expected input files in data/raw/:
    - reviews.csv   : reviewerID, asin, overall, unixReviewTime,
                      reviewText, summary, reviewerName, helpful
    - products.csv  : asin, title, description, categories, price
      (optional — if absent, a minimal product table is derived from reviews)

Output files written to data/processed/:
    - interactions.parquet : user_id, product_id, rating, timestamp
    - products.parquet     : product_id, title, description, category, price
    - users.parquet        : user_id, total_reviews, avg_rating
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent          # data/
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_reviews(path: Path) -> pd.DataFrame:
    """Load the reviews CSV and normalise column names.

    Parameters
    ----------
    path:
        Path to the raw reviews CSV file.

    Returns
    -------
    pd.DataFrame
        Raw reviews with at minimum the columns:
        ``user_id``, ``product_id``, ``rating``, ``timestamp``.
    """
    df = pd.read_csv(path, low_memory=False)

    # Map common Amazon dataset column names → internal names
    rename_map: dict[str, str] = {
        "reviewerID": "user_id",
        "asin":       "product_id",
        "overall":    "rating",
        "unixReviewTime": "timestamp",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns exist
    required = {"user_id", "product_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reviews.csv is missing required columns: {missing}")

    return df


def load_products(path: Path) -> pd.DataFrame | None:
    """Load the optional products metadata CSV.

    Parameters
    ----------
    path:
        Path to the raw products CSV file.

    Returns
    -------
    pd.DataFrame or None
        Product metadata, or *None* if the file does not exist.
    """
    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)

    rename_map: dict[str, str] = {
        "asin":        "product_id",
        "categories":  "category",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, subset: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows based on *subset* columns.

    Returns
    -------
    (cleaned_df, n_dropped)
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    return df, before - len(df)


def drop_nulls(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop rows where any of *columns* is null.

    Returns
    -------
    (cleaned_df, n_dropped)
    """
    before = len(df)
    df = df.dropna(subset=columns)
    return df, before - len(df)


def filter_cold_start(
    df: pd.DataFrame,
    min_user_reviews: int = 5,
    min_product_reviews: int = 10,
) -> tuple[pd.DataFrame, int, int]:
    """Remove cold-start users and products iteratively until stable.

    Parameters
    ----------
    df:
        Interactions DataFrame with ``user_id`` and ``product_id`` columns.
    min_user_reviews:
        Minimum number of reviews a user must have.
    min_product_reviews:
        Minimum number of reviews a product must have.

    Returns
    -------
    (filtered_df, n_users_removed, n_products_removed)
    """
    users_removed_total = 0
    products_removed_total = 0

    while True:
        prev_len = len(df)

        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        before_u = len(df)
        df = df[df["user_id"].isin(valid_users)]
        users_removed_total += before_u - len(df)

        # Filter products
        product_counts = df["product_id"].value_counts()
        valid_products = product_counts[product_counts >= min_product_reviews].index
        before_p = len(df)
        df = df[df["product_id"].isin(valid_products)]
        products_removed_total += before_p - len(df)

        # Stop when nothing changes in this pass
        if len(df) == prev_len:
            break

    return df, users_removed_total, products_removed_total


# ---------------------------------------------------------------------------
# Artefact builders
# ---------------------------------------------------------------------------

def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and type-cast the interactions artefact.

    Parameters
    ----------
    df:
        Cleaned reviews DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id`` (str), ``product_id`` (str),
        ``rating`` (float32), ``timestamp`` (int64, Unix seconds).
    """
    cols = ["user_id", "product_id", "rating"]
    if "timestamp" in df.columns:
        cols.append("timestamp")

    interactions = df[cols].copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["product_id"] = interactions["product_id"].astype(str)
    interactions["rating"] = interactions["rating"].astype("float32")

    if "timestamp" in interactions.columns:
        interactions["timestamp"] = pd.to_numeric(
            interactions["timestamp"], errors="coerce"
        ).astype("Int64")
    else:
        interactions["timestamp"] = pd.NA

    return interactions


def build_products(
    reviews_df: pd.DataFrame,
    products_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the products artefact.

    If *products_meta* is available, it is joined with the active product IDs
    from *reviews_df*. Otherwise a minimal table is derived from the reviews.

    Parameters
    ----------
    reviews_df:
        Cleaned interactions DataFrame (provides the authoritative product set).
    products_meta:
        Optional product metadata loaded from ``data/raw/products.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``product_id``, ``title``, ``description``,
        ``category``, ``price``.
    """
    active_ids = reviews_df["product_id"].unique()

    if products_meta is not None:
        products_meta["product_id"] = products_meta["product_id"].astype(str)
        products = products_meta[products_meta["product_id"].isin(active_ids)].copy()
    else:
        products = pd.DataFrame({"product_id": active_ids})

    for col in ("title", "description", "category", "price"):
        if col not in products.columns:
            products[col] = pd.NA

    return products[["product_id", "title", "description", "category", "price"]].copy()


def build_users(interactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user statistics.

    Parameters
    ----------
    interactions:
        Cleaned interactions DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id`` (str), ``total_reviews`` (int64),
        ``avg_rating`` (float32).
    """
    users = (
        interactions.groupby("user_id", as_index=False)
        .agg(
            total_reviews=("rating", "count"),
            avg_rating=("rating", "mean"),
        )
    )
    users["total_reviews"] = users["total_reviews"].astype("int64")
    users["avg_rating"] = users["avg_rating"].astype("float32")
    return users


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def sparsity(n_users: int, n_items: int, n_interactions: int) -> float:
    """Return the sparsity of the user-item matrix as a fraction in [0, 1].

    Sparsity = 1 − (observed / possible)
    """
    possible = n_users * n_items
    if possible == 0:
        return 1.0
    return 1.0 - n_interactions / possible


def print_quality_report(
    n_raw: int,
    n_after_dedup: int,
    n_after_nulls: int,
    n_after_rating_filter: int,
    n_after_cold_start: int,
    interactions: pd.DataFrame,
) -> None:
    """Print a human-readable data-quality report to stdout.

    Parameters
    ----------
    n_raw:
        Record count in the original CSV.
    n_after_dedup:
        Record count after duplicate removal.
    n_after_nulls:
        Record count after null removal.
    n_after_rating_filter:
        Record count after invalid-rating removal (rating not in [1, 5]).
    n_after_cold_start:
        Record count after cold-start filtering.
    interactions:
        Final interactions DataFrame.
    """
    sep = "─" * 60

    print(f"\n{sep}")
    print("  SmartRec — Data Quality Report")
    print(sep)

    print("\n[Record counts]")
    print(f"  Raw records          : {n_raw:>10,}")
    print(f"  After deduplication  : {n_after_dedup:>10,}  (dropped {n_raw - n_after_dedup:,})")
    print(f"  After null removal   : {n_after_nulls:>10,}  (dropped {n_after_dedup - n_after_nulls:,})")
    print(f"  After rating filter  : {n_after_rating_filter:>10,}  (dropped {n_after_nulls - n_after_rating_filter:,})")
    print(f"  After cold-start     : {n_after_cold_start:>10,}  (dropped {n_after_rating_filter - n_after_cold_start:,})")

    n_users = interactions["user_id"].nunique()
    n_items = interactions["product_id"].nunique()
    n_interactions = len(interactions)

    print("\n[Matrix dimensions]")
    print(f"  Unique users         : {n_users:>10,}")
    print(f"  Unique products      : {n_items:>10,}")
    print(f"  Interactions         : {n_interactions:>10,}")
    print(f"  Sparsity             : {sparsity(n_users, n_items, n_interactions):>10.4%}")

    print("\n[Rating distribution]")
    dist = interactions["rating"].value_counts().sort_index()
    for rating, count in dist.items():
        bar = "█" * int(count / n_interactions * 40)
        print(f"  {rating:.1f}  {bar:<40}  {count:,} ({count / n_interactions:.1%})")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    min_user_reviews: int = 5,
    min_product_reviews: int = 10,
) -> None:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    min_user_reviews:
        Minimum reviews per user to be retained.
    min_product_reviews:
        Minimum reviews per product to be retained.
    """
    reviews_path = RAW_DIR / "reviews.csv"
    products_path = RAW_DIR / "products.csv"

    if not reviews_path.exists():
        print(
            f"[ERROR] Expected reviews file not found: {reviews_path}\n"
            "        Place the Amazon Electronics Reviews CSV at data/raw/reviews.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading reviews from {reviews_path} …")
    df = load_reviews(reviews_path)
    n_raw = len(df)

    print("Loading product metadata …")
    products_meta = load_products(products_path)
    if products_meta is None:
        print("  products.csv not found — product table will be minimal.")

    # --- Deduplication --------------------------------------------------------
    print("Removing duplicates …")
    df, n_dup = remove_duplicates(df, subset=["user_id", "product_id"])
    n_after_dedup = len(df)

    # --- Null removal ---------------------------------------------------------
    print("Dropping rows with null required fields …")
    df, n_null = drop_nulls(df, columns=["user_id", "product_id", "rating"])
    n_after_nulls = len(df)

    # --- Rating validity ------------------------------------------------------
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5)]
    n_after_rating_filter = len(df)

    # --- Cold-start filtering -------------------------------------------------
    print(
        f"Filtering cold-start (min_user_reviews={min_user_reviews}, "
        f"min_product_reviews={min_product_reviews}) …"
    )
    df, _, _ = filter_cold_start(df, min_user_reviews, min_product_reviews)
    n_after_cold_start = len(df)

    # --- Build artefacts ------------------------------------------------------
    print("Building artefacts …")
    interactions = build_interactions(df)
    products = build_products(df, products_meta)
    users = build_users(interactions)

    # --- Save -----------------------------------------------------------------
    interactions_path = PROCESSED_DIR / "interactions.parquet"
    products_path_out = PROCESSED_DIR / "products.parquet"
    users_path_out = PROCESSED_DIR / "users.parquet"

    interactions.to_parquet(interactions_path, index=False, engine="pyarrow")
    products.to_parquet(products_path_out, index=False, engine="pyarrow")
    users.to_parquet(users_path_out, index=False, engine="pyarrow")

    print(f"  Saved → {interactions_path}")
    print(f"  Saved → {products_path_out}")
    print(f"  Saved → {users_path_out}")

    # --- Report ---------------------------------------------------------------
    print_quality_report(
        n_raw=n_raw,
        n_after_dedup=n_after_dedup,
        n_after_nulls=n_after_nulls,
        n_after_rating_filter=n_after_rating_filter,
        n_after_cold_start=n_after_cold_start,
        interactions=interactions,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartRec data preprocessing pipeline")
    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=5,
        help="Minimum reviews per user (default: 5)",
    )
    parser.add_argument(
        "--min-product-reviews",
        type=int,
        default=10,
        help="Minimum reviews per product (default: 10)",
    )
    args = parser.parse_args()

    run(
        min_user_reviews=args.min_user_reviews,
        min_product_reviews=args.min_product_reviews,
    )
