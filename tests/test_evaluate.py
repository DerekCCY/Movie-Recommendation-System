"""
Tests for ml_pipeline.evaluate module
-------------------------------------
Covers:
- evaluate_rmse
- evaluate_precision_at_k
- evaluate_recall_at_k
- measure_inference_time
- evaluate_recommendation_diversity
- generate_evaluation_report
"""

import pytest
import pandas as pd
import numpy as np
from ml_pipeline import evaluate


# -------------------------------------------------------------------
# Dummy Model used for all metric tests
# -------------------------------------------------------------------

class DummyModel:
    """Simple mock recommendation model for evaluation tests."""

    def __init__(self):
        self.user_ids = ["u1", "u2", "u3"]
        # pretend every predict call returns same 5 movies
        self.recs = ["m1", "m2", "m3", "m4", "m5"]

    # Used by evaluate_rmse
    def predict_rating(self, user_id, movie_id):
        # deterministic fake prediction: rating = 4.0
        return 4.0

    # Used by precision/recall and inference time
    def predict(self, user_id, n_recommendations=5):
        return self.recs[:n_recommendations]

    def get_model_info(self):
        return {
            "algorithm": "DummySVD",
            "total_users": 3,
            "total_movies": 5,
            "n_factors": 10
        }


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def simple_test_df():
    """Small test DataFrame with 3 users × 3 movies."""
    data = {
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "movie_id": ["m1", "m2", "m2", "m3", "m4", "m5"],
        "rating": [5.0, 4.0, 2.0, 5.0, 3.0, 1.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_model():
    return DummyModel()


# -------------------------------------------------------------------
# RMSE
# -------------------------------------------------------------------

def test_evaluate_rmse(dummy_model, simple_test_df):
    """RMSE should calculate correctly from dummy predictions."""
    rmse = evaluate.evaluate_rmse(dummy_model, simple_test_df)
    # Actual ratings vary, predicted is fixed 4.0
    # So RMSE > 0 but finite
    assert isinstance(rmse, float)
    assert rmse >= 0.0
    assert rmse < 3.0  # with this data it's small


# -------------------------------------------------------------------
# Precision@K
# -------------------------------------------------------------------

def test_evaluate_precision_at_k(dummy_model, simple_test_df):
    """Precision@K should return a float between 0 and 1."""
    precision = evaluate.evaluate_precision_at_k(dummy_model, simple_test_df, k=3)
    assert isinstance(precision, float)
    assert 0.0 <= precision <= 1.0


def test_precision_handles_no_relevant_items(dummy_model):
    """Precision@K should return 0.0 when no ratings >= threshold."""
    df = pd.DataFrame({
        "user_id": ["u1", "u1"],
        "movie_id": ["m1", "m2"],
        "rating": [2.0, 1.0]  # all below 3.5 threshold
    })
    result = evaluate.evaluate_precision_at_k(dummy_model, df, k=5)
    assert result == 0.0


# -------------------------------------------------------------------
# Recall@K
# -------------------------------------------------------------------

def test_evaluate_recall_at_k(dummy_model, simple_test_df):
    """Recall@K should return a float between 0 and 1."""
    recall = evaluate.evaluate_recall_at_k(dummy_model, simple_test_df, k=3)
    assert isinstance(recall, float)
    assert 0.0 <= recall <= 1.0


def test_recall_handles_no_relevant_items(dummy_model):
    """Recall@K should return 0.0 when no ratings >= threshold."""
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "rating": [2.0, 1.0]
    })
    result = evaluate.evaluate_recall_at_k(dummy_model, df, k=3)
    assert result == 0.0


# -------------------------------------------------------------------
# Inference time measurement
# -------------------------------------------------------------------

def test_measure_inference_time(dummy_model):
    """Should return valid performance stats with correct keys."""
    stats = evaluate.measure_inference_time(dummy_model, n_samples=5, n_recommendations=2)
    expected_keys = {"mean_time_ms", "p95_time_ms", "requests_per_second"}
    assert expected_keys.issubset(stats.keys())
    assert all(isinstance(v, float) for v in stats.values())
    assert stats["mean_time_ms"] >= 0.0


# -------------------------------------------------------------------
# Recommendation diversity
# -------------------------------------------------------------------

def test_evaluate_recommendation_diversity():
    """Diversity = (# of unique items) / total items (here we just check count)."""
    rec_lists = [
        ["m1", "m2", "m3"],
        ["m3", "m4"],
        ["m1", "m5"],
    ]
    diversity = evaluate.evaluate_recommendation_diversity(rec_lists)
    # Should return number of unique items
    assert diversity == 5  # m1, m2, m3, m4, m5


# -------------------------------------------------------------------
# Evaluation report
# -------------------------------------------------------------------

def test_generate_evaluation_report(dummy_model, simple_test_df):
    """Integration test for generate_evaluation_report()."""
    config = {"k_values": [3], "n_inference_samples": 2}
    report = evaluate.generate_evaluation_report(dummy_model, simple_test_df, config)

    # Verify report contains all expected metrics
    expected_sections = {
        "rmse", "precision@k", "recall@k", "inference_time", "model_info"
    }
    assert expected_sections.issubset(report.keys())
    assert isinstance(report["rmse"], float)
    assert isinstance(report["precision@k"], dict)
    assert isinstance(report["recall@k"], dict)
    assert isinstance(report["inference_time"], dict)
    assert "mean_time_ms" in report["inference_time"]
    assert "algorithm" in report["model_info"]
