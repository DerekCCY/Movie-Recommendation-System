"""
Silver Layer Cleaning Module
============================

Builds clean user-movie interactions (Silver layer) from Bronze parquet files.

Steps:
1. Load Bronze parquet files (watch_events, rating_events)
2. Aggregate watch minutes
3. Merge ratings (if available)
4. Validate schema and clean data
5. Deduplicate interactions
6. Filter sparse users/movies
7. Save to Silver layer

Output:
    data/silver/interactions.parquet
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd

from ..config import BRONZE_DIR, SILVER_DIR, PREPROCESSING_CONFIG

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Utility: Drop duplicates keeping latest timestamp
# ---------------------------------------------------------------------
def drop_duplicates_keep_latest(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """Drop duplicates keeping the latest row by timestamp if available."""
    if "ts" in df.columns:
        return df.sort_values("ts").drop_duplicates(subset=subset, keep="last")
    return df.drop_duplicates(subset=subset, keep="last")


# ---------------------------------------------------------------------
# Utility: Convert watch minutes to approximate rating (optional heuristic)
# ---------------------------------------------------------------------
def convert_watch_minutes_to_rating(minutes: int) -> int:
    """Convert implicit watch time to 1–5 rating scale (heuristic)."""
    if minutes <= 0:
        return 1
    elif minutes <= 5:
        return 2
    elif minutes <= 20:
        return 2
    elif minutes <= 45:
        return 3
    elif minutes <= 90:
        return 4
    else:
        return 5


# ---------------------------------------------------------------------
# Main: Clean interactions to build Silver layer
# ---------------------------------------------------------------------
def clean_interactions_for_silver(
    bronze_dir: str = str(BRONZE_DIR),
    output_path: str = str(SILVER_DIR / "interactions.parquet"),
    config: Optional[dict] = None,
    report_quality: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Build clean user-movie interactions table (Silver layer).

    Returns:
        (cleaned_df, quality_report) if report_quality=True
        cleaned_df only otherwise
    """
    cfg = config or PREPROCESSING_CONFIG
    logger.info(f"🔄 Starting Silver cleaning from {bronze_dir}")

    # ------------------------------------------------------------------
    # 1️⃣ Load Bronze parquet files
    # ------------------------------------------------------------------
    watch_path = Path(bronze_dir) / "watch_events"
    rate_path = Path(bronze_dir) / "rating_events"

    watch_files = list(watch_path.rglob("*.parquet"))
    if not watch_files:
        raise FileNotFoundError("❌ No watch parquet files found in bronze/watch_events/")
    watch_df = pd.concat([pd.read_parquet(f) for f in watch_files], ignore_index=True)
    logger.info(f"🎬 Loaded {len(watch_df)} watch events")

    # Aggregate and keep the latest timestamp
    watch_df = (
        pd.concat([pd.read_parquet(f) for f in watch_files], ignore_index=True)
        .sort_values("ts")
        .groupby(["user_id", "movie_id"], as_index=False)
        .agg(
            watch_minutes=("minute", "count"),
            timestamp=("ts", "last"),   # keep the last interaction timestamp
        )
    )

    rating_df = pd.DataFrame()
    if rate_path.exists():
        rating_files = list(rate_path.rglob("*.parquet"))
        if rating_files:
            rating_df = pd.concat([pd.read_parquet(f) for f in rating_files], ignore_index=True)
            logger.info(f"⭐ Loaded {len(rating_df)} rating events")

    # ------------------------------------------------------------------
    # 2️⃣ Merge watch & rating data
    # ------------------------------------------------------------------
    if not rating_df.empty:
        interactions = watch_df.merge(
            rating_df[["user_id", "movie_id", "rating"]],
            on=["user_id", "movie_id"],
            how="left",
        )
    else:
        interactions = watch_df
        interactions["rating"] = interactions["watch_minutes"].apply(convert_watch_minutes_to_rating)

    logger.info(f"🧩 Merged interactions: {len(interactions)} rows")

    # ------------------------------------------------------------------
    # 3️⃣ Validate schema & types
    # ------------------------------------------------------------------
    before = len(interactions)
    interactions = interactions.dropna(subset=["user_id", "movie_id"])
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["movie_id"] = interactions["movie_id"].astype(str)
    interactions["watch_minutes"] = pd.to_numeric(interactions["watch_minutes"], errors="coerce").fillna(0).astype(int)

    min_r, max_r = cfg.get("min_rating", 1), cfg.get("max_rating", 5)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce").fillna(min_r)
    interactions["rating"] = interactions["rating"].clip(min_r, max_r).astype(int)

    dropped_invalid = before - len(interactions)
    if dropped_invalid > 0:
        logger.warning(f"⚠️ Dropped {dropped_invalid} invalid rows with missing IDs or bad values")

    # ------------------------------------------------------------------
    # 4️⃣ Deduplicate
    # ------------------------------------------------------------------
    before = len(interactions)
    interactions = drop_duplicates_keep_latest(interactions, ["user_id", "movie_id"])
    logger.info(f"🧹 Removed {before - len(interactions)} duplicate interactions")

    # ------------------------------------------------------------------
    # 5️⃣ Sparsity filtering
    # ------------------------------------------------------------------
    before = len(interactions)
    min_user_inter = cfg.get("min_user_interactions", 0)
    min_movie_inter = cfg.get("min_movie_interactions", 0)

    if min_user_inter > 0:
        user_counts = interactions["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_inter].index
        interactions = interactions[interactions["user_id"].isin(valid_users)]

    if min_movie_inter > 0:
        movie_counts = interactions["movie_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_inter].index
        interactions = interactions[interactions["movie_id"].isin(valid_movies)]

    filtered_out = before - len(interactions)
    if filtered_out > 0:
        logger.info(f"⚠️ Removed {filtered_out} interactions due to sparsity filters")

    # ------------------------------------------------------------------
    # 6️⃣ Save to Silver layer
    # ------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(output_path, index=False)
    logger.info(f"💾 Saved Silver dataset → {output_path}")

    # ------------------------------------------------------------------
    # 7️⃣ Generate quality report
    # ------------------------------------------------------------------
    quality_report = None
    if report_quality:
        n_users = interactions["user_id"].nunique()
        n_movies = interactions["movie_id"].nunique()
        n_rows = len(interactions)
        sparsity = 1 - (n_rows / (n_users * n_movies)) if n_users and n_movies else 0

        quality_report = {
            "initial_rows": int(before),
            "final_rows": int(n_rows),
            "rows_removed": int(before - n_rows),
            "data_retention_rate": float(n_rows / before if before > 0 else 0),
            "unique_users": int(n_users),
            "unique_movies": int(n_movies),
            "sparsity": float(sparsity),
            "avg_rating": float(interactions["rating"].mean()),
            "rating_distribution": {
                int(k): int(v)
                for k, v in interactions["rating"].value_counts().sort_index().to_dict().items()
            },
        }

        logger.info("=" * 60)
        logger.info("SILVER DATA QUALITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Rows before cleaning: {before}")
        logger.info(f"Rows after cleaning: {n_rows}")
        logger.info(f"Data retention: {quality_report['data_retention_rate']*100:.1f}%")
        logger.info(f"Unique users: {n_users}, movies: {n_movies}")
        logger.info(f"Sparsity: {sparsity:.4f}")
        logger.info(f"Average rating: {quality_report['avg_rating']:.2f}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 8️⃣ Return result(s)
    # ------------------------------------------------------------------
    if report_quality:
        return interactions, quality_report
    else:
        return interactions