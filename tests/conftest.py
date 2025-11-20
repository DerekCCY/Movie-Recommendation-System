"""
Pytest configuration and shared fixtures

This file contains fixtures that are available to all test files.
"""

import pytest
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Path settings
# ---------------------------------------------------
@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test fixtures directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_interactions_path(test_data_dir):
    """Path to sample interactions parquet file"""
    return test_data_dir / "sample_data.parquet"

# ---------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------

@pytest.fixture(scope="session")
def sample_interactions_df(sample_interactions_path):
    """
    Load the real sample interactions parquet file.
    This fixture should be used for integration / feature-level tests.
    """
    try:
        return pd.read_parquet(sample_interactions_path)
    except FileNotFoundError:
        # fallback if parquet not present (use dummy df)
        return pd.DataFrame({
            "user_id": ["user_1", "user_2", "user_3"],
            "movie_id": ["m1", "m2", "m3"],
            "rating": [5, 3, 4],
            "watch_minutes": [90, 45, 60],
        })


@pytest.fixture
def tiny_interactions_df():
    """
    Tiny interactions DataFrame (3 users, 3 movies) for unit tests
    """
    return pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u3'],
        'movie_id': ['m1', 'm2', 'm1', 'm3', 'm2'],
        'rating': [5, 3, 4, 5, 2]
    })
    
# ---------------------------------------------------
# Utility fixture: temporary directory
# ---------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """
    Create a temporary output directory for tests that write files.
    Automatically cleaned up after test.
    """
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    return out_dir

# ---------------------------------------------------
# Hybrid Model Fixtures
# ---------------------------------------------------

@pytest.fixture
def sample_interactions_with_metadata():
    """
    Sample interactions DataFrame with metadata for hybrid model testing.

    Includes: user_id, movie_id, rating, watch_minutes, runtime,
              genres, vote_average, popularity

    50 rows, 5 users, 10 movies with realistic metadata
    """
    import numpy as np

    # 5 users, 10 movies, ~50 interactions
    data = {
        'user_id': ['u1']*10 + ['u2']*10 + ['u3']*10 + ['u4']*10 + ['u5']*10,
        'movie_id': ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10'] * 5,
        'rating': [5, 4, 3, 5, 4, 2, 5, 3, 4, 5,
                   4, 5, 2, 4, 3, 5, 4, 2, 3, 4,
                   3, 4, 5, 3, 4, 1, 5, 4, 5, 3,
                   5, 3, 4, 5, 2, 4, 3, 5, 4, 2,
                   4, 5, 3, 4, 5, 3, 2, 4, 3, 5],
        'watch_minutes': [120, 90, 60, 110, 100, 30, 115, 75, 95, 125,
                          100, 110, 45, 95, 80, 120, 105, 40, 70, 90,
                          85, 100, 115, 75, 95, 20, 120, 100, 110, 80,
                          115, 85, 95, 120, 50, 100, 75, 115, 95, 45,
                          95, 120, 80, 100, 115, 85, 55, 95, 75, 120],
        'runtime': [130, 110, 95, 125, 115, 100, 130, 105, 110, 135,
                    130, 110, 95, 125, 115, 100, 130, 105, 110, 135,
                    130, 110, 95, 125, 115, 100, 130, 105, 110, 135,
                    130, 110, 95, 125, 115, 100, 130, 105, 110, 135,
                    130, 110, 95, 125, 115, 100, 130, 105, 110, 135],
        'genres': [
            "['Action', 'Sci-Fi']", "['Drama', 'Romance']", "['Comedy']",
            "['Action', 'Adventure']", "['Drama']", "['Horror', 'Thriller']",
            "['Sci-Fi', 'Thriller']", "['Comedy', 'Romance']", "['Action']",
            "['Drama', 'Sci-Fi']"
        ] * 5,
        'vote_average': [7.5, 8.2, 6.5, 7.8, 8.0, 5.5, 7.9, 6.8, 7.2, 8.1,
                         7.5, 8.2, 6.5, 7.8, 8.0, 5.5, 7.9, 6.8, 7.2, 8.1,
                         7.5, 8.2, 6.5, 7.8, 8.0, 5.5, 7.9, 6.8, 7.2, 8.1,
                         7.5, 8.2, 6.5, 7.8, 8.0, 5.5, 7.9, 6.8, 7.2, 8.1,
                         7.5, 8.2, 6.5, 7.8, 8.0, 5.5, 7.9, 6.8, 7.2, 8.1],
        'popularity': [150.5, 200.3, 80.2, 175.6, 190.1, 60.5, 165.3, 95.4, 140.2, 210.5,
                       150.5, 200.3, 80.2, 175.6, 190.1, 60.5, 165.3, 95.4, 140.2, 210.5,
                       150.5, 200.3, 80.2, 175.6, 190.1, 60.5, 165.3, 95.4, 140.2, 210.5,
                       150.5, 200.3, 80.2, 175.6, 190.1, 60.5, 165.3, 95.4, 140.2, 210.5,
                       150.5, 200.3, 80.2, 175.6, 190.1, 60.5, 165.3, 95.4, 140.2, 210.5],
    }

    df = pd.DataFrame(data)

    # Add some NaN values to test robustness
    df.loc[0, 'vote_average'] = np.nan
    df.loc[5, 'popularity'] = np.nan
    df.loc[10, 'runtime'] = np.nan

    return df


@pytest.fixture
def hybrid_train_val_split(sample_interactions_with_metadata):
    """
    Pre-split train/val data for hybrid model testing (30/10 split).

    Returns:
        Tuple of (train_df, val_df)
    """
    df = sample_interactions_with_metadata

    # Simple 80/20 split (40 train, 10 val)
    train_df = df.iloc[:40].copy()
    val_df = df.iloc[40:].copy()

    return train_df, val_df


@pytest.fixture
def trained_hybrid_model(sample_interactions_with_metadata):
    """
    Pre-fitted PersonalizedHybridRecommender for quick tests.

    Trained on sample_interactions_with_metadata with small n_factors=10.
    """
    from ml_pipeline.model import PersonalizedHybridRecommender

    df = sample_interactions_with_metadata

    # Split data
    train_df = df.iloc[:40].copy()
    val_df = df.iloc[40:].copy()

    # Train model with small config for speed
    model = PersonalizedHybridRecommender(
        n_factors=10,
        learning_rate=0.02,
        regularization=0.01,
        content_weight=0.3
    )

    model.fit(
        train_df=train_df,
        val_df=val_df,
        interactions_df=df,
        n_epochs=5  # Fast training for tests
    )

    return model