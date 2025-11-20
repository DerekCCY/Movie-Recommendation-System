from ml_pipeline.data_io.bronze_ingest import run_kafka_ingest_continuous



'''Bronze'''
#run_kafka_ingest_continuous(
#    event_types= None,  # 可改 None 表示全部
#    flush_every=2000,                 # 每 2000 筆寫一次
#    flush_secs=30,                    # 或每 30 秒寫一次
#    partition_by_hour=True            # 自動建 dt=YYYY-MM-DD/hour=HH/
#)

'''Silver'''

#from ml_pipeline.data_io.silver_cleaning import clean_interactions_for_silver
#
#df_silver = clean_interactions_for_silver()
#print(df_silver.head())

'''Gold'''
#from ml_pipeline.data_io.gold_enrichment import load_enriched_interactions
#df_gold = load_enriched_interactions()
#print(df_gold.info())
"""
Feature Engineering Pipeline Entrypoint

Loads cleaned Gold layer interactions and produces:
- Sparse user-item rating matrix
- Index mappings (user/movie)
- Popularity and bias features
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import save_npz

from ..config import GOLD_DIR, FEATURES_DIR
from ml_pipeline.feature_engineering import (
    filter_sparse_interactions,
    build_user_item_matrix,
    calculate_popular_movies,
    calculate_global_statistics,
)

def run_feature_engineering(
    gold_path: str = str(GOLD_DIR / "interactions_enriched.parquet"),
    output_dir: str = str(FEATURES_DIR)
) -> None:
    """
    Run end-to-end feature engineering pipeline.

    Args:
        gold_path: Path to Gold dataset parquet
        output_dir: Directory to save outputs
    """
    print(f"📥 Loading Gold data from {gold_path}")
    df = pd.read_parquet(gold_path)

    print(f"💡 Total interactions: {len(df)}")

    # 1️⃣ Filter sparse users/movies
    df_filtered = filter_sparse_interactions(df, min_user_interactions=5, min_item_interactions=3)
    print(f"✅ Filtered interactions: {len(df_filtered)}")

    # 2️⃣ Build sparse user-item matrix + mappings
    matrix, mappings = build_user_item_matrix(df_filtered)
    print(f"🧮 Built user-item matrix: shape = {matrix.shape}")

    # 3️⃣ Calculate popularity-based features
    top_movies = calculate_popular_movies(df_filtered, n_top=50)
    print(f"🎬 Top 5 popular movies: {top_movies[:5]}")

    # 4️⃣ Compute bias terms
    stats = calculate_global_statistics(df_filtered, mappings)
    print(f"📊 Global mean rating: {stats['global_mean']:.3f}")

    # 5️⃣ Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(output_dir / "user_item_matrix.npz", matrix)
    with open(output_dir / "mappings.json", "w") as f:
        json.dump({k: list(v.keys()) if isinstance(v, dict) else v for k, v in mappings.items()}, f, indent=2)
    with open(output_dir / "global_stats.json", "w") as f:
        json.dump(
            {
                "global_mean": stats["global_mean"],
                "user_biases_mean": float(np.mean(stats["user_biases"])),
                "item_biases_mean": float(np.mean(stats["item_biases"]))
            },
            f,
            indent=2
        )

    print(f"💾 Saved feature artifacts to {output_dir}")
    print("🎉 Feature engineering complete!")


if __name__ == "__main__":
    run_feature_engineering()

