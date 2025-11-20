"""
Quick test script for hybrid model training on small sample.
Tests for NaN issues in ~5-10 minutes instead of 3 hours.

Usage:
    python scripts/test_hybrid_small.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add ml_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.model import PersonalizedHybridRecommender
from ml_pipeline.config import HYBRID_CONFIG


def main():
    print("=" * 60)
    print("QUICK HYBRID MODEL TEST (5K SAMPLE)")
    print("=" * 60)

    # Load enriched data
    data_path = Path.home() / "data" / "enriched" / "interactions_with_meta.parquet"
    if not data_path.exists():
        # Fallback to project data
        data_path = Path(__file__).parent.parent / "data" / "silver" / "interactions.parquet"

    print(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Total rows: {len(df):,}")

    # Sample 5k rows
    sample_size = 5000
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f"Using sample: {len(df_sample):,} rows")

    # Split train/val
    split_idx = int(len(df_sample) * 0.8)
    train_df = df_sample.iloc[:split_idx].copy()
    val_df = df_sample.iloc[split_idx:].copy()

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Initialize model
    print("\nInitializing PersonalizedHybridRecommender...")
    model = PersonalizedHybridRecommender(
        n_factors=HYBRID_CONFIG.get('n_factors', 100),
        learning_rate=HYBRID_CONFIG.get('learning_rate', 0.015),
        regularization=HYBRID_CONFIG.get('regularization', 0.005),
        content_weight=HYBRID_CONFIG.get('content_weight', 0.25)
    )

    # Train for just 5 epochs (quick test)
    print("\nTraining for 5 epochs (quick test)...")
    try:
        result = model.fit(
            train_df=train_df,
            val_df=val_df,
            interactions_df=df_sample,
            n_epochs=5,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("✅ SUCCESS! No NaN errors detected")
        print("=" * 60)
        print(f"Training time: {result:.2f}s")

        # Test prediction
        sample_user = train_df.iloc[0]['user_id']
        sample_movie = train_df.iloc[0]['movie_id']
        prediction = model.predict_rating(sample_user, sample_movie)

        print(f"\nTest prediction: {prediction:.2f}")

        if pd.notna(prediction) and not pd.isnull(prediction):
            print("✅ Prediction is valid (not NaN)")
            print("\n🎉 Model is ready for full training!")
            return 0
        else:
            print("❌ Prediction is NaN - model has issues")
            return 1

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TRAINING FAILED")
        print("=" * 60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
