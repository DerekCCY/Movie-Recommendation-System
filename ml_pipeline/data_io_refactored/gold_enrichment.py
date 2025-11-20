import os
from pathlib import Path
import pandas as pd

from ..config import SILVER_DIR, GOLD_DIR, API_CACHE_PATH  
from .api_client import MovieAPIClient

def _validate_interactions_schema(df: pd.DataFrame) -> None:
    expected = {"user_id", "movie_id"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def load_enriched_interactions(
    interactions_path: str = str(SILVER_DIR / "interactions.parquet"),
    cache_path: str = str(API_CACHE_PATH),
    output_path: str = str(GOLD_DIR / "interactions_enriched.parquet"),
) -> pd.DataFrame:
    """
    Load silver interactions, enrich with cached movie metadata, and save to gold layer.
    """
    print(f"🔹 Loading base interactions from {interactions_path}")
    df = pd.read_parquet(interactions_path)
    _validate_interactions_schema(df)

    client = MovieAPIClient(cache_path=cache_path)
    if not client.cache:
        raise FileNotFoundError(
            f"No movie cache found for enrichment at {cache_path}. "
            "Populate cache with MovieAPIClient.save_cache() or provide a prebuilt cache."
        )

    # only movie keys (exclude user_* cache)
    movie_items = {k: v for k, v in client.cache.items() if not str(k).startswith("user_")}
    movie_meta_df = pd.DataFrame.from_dict(movie_items, orient="index").reset_index()
    movie_meta_df.rename(columns={"index": "movie_id"}, inplace=True)
    print(f"🎬 Loaded {len(movie_meta_df)} cached movie metadata entries")

    enriched = df.merge(movie_meta_df, on="movie_id", how="left")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    print(f"✅ Enriched interactions saved to {output_path} ({len(enriched)} rows)")
    return enriched
