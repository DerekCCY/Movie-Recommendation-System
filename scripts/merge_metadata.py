"""
M2 Data Enrichment Script
Merges interactions dataset with pre-cached movie metadata from Derek's API cache.

This script enriches the gold-level interactions dataset with movie metadata including:
- Movie details (title, year, runtime, budget)
- Popularity metrics (vote_average, vote_count, popularity)
- Categories (genres, spoken_languages, content_rating)

Usage:
    python merge_metadata.py

The script expects:
- Input: data/gold/interactions.parquet (from preprocessing pipeline)
- Cache: data/cache/movie_api_cache.json (Derek's pre-cached metadata)
- Output: data/enriched/interactions_with_meta.parquet

For M2, we use Derek's pre-cached metadata to avoid live API calls during training.
"""

import pandas as pd
import os
import json
from pathlib import Path


def enrich_interactions_with_metadata(
    interactions_path: str,
    output_path: str = "data/enriched/interactions_with_meta.parquet",
    cache_path: str = "data/cache/movie_api_cache.json"
) -> pd.DataFrame:
    """
    Load interactions parquet, enrich with movie metadata from cache,
    and save to a new parquet file.

    Args:
        interactions_path: Path to base interactions parquet file
        output_path: Path to save enriched data
        cache_path: Path for API cache (Derek's pre-cached metadata)

    Returns:
        pd.DataFrame: Enriched dataframe with 27 features
    """
    print(f"🔹 Loading base interactions from {interactions_path}")
    df = pd.read_parquet(interactions_path)
    print(f"Loaded {len(df)} interactions, {df['user_id'].nunique()} users, {df['movie_id'].nunique()} movies")

    # Load pre-cached movie metadata
    movie_ids = df['movie_id'].unique().tolist()
    print(f"Loading metadata for {len(movie_ids)} movies from cache...")

    with open(cache_path, "r") as f:
        cache_data = json.load(f)

    movie_meta_df = pd.DataFrame.from_dict(cache_data, orient='index').reset_index()
    movie_meta_df.rename(columns={"index": "movie_id"}, inplace=True)
    print(f"Loaded {len(movie_meta_df)} valid movie metadata entries from cache")

    # Merge movie metadata
    df = df.merge(movie_meta_df, on="movie_id", how="left")

    # Save output
    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Enriched data saved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    return df


if __name__ == "__main__":
    enriched_df = enrich_interactions_with_metadata(
        interactions_path="data/gold/interactions.parquet"
    )
    print("\nSample enriched data:")
    print(enriched_df.head())
