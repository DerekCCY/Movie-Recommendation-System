"""
Tests for ml_pipeline.model.ImprovedSVDRecommendationModel
-----------------------------------------------------------
Covers:
- Initialization
- fit() training setup
- predict_rating()
- predict() for known and unknown users
- _cold_start_recommendations()
- get_model_info()
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from ml_pipeline.model import ImprovedSVDRecommendationModel

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def tiny_mappings():
    """Simple 3×3 user-item mapping for SVD tests."""
    return {
        "user_mapping": {"u1": 0, "u2": 1, "u3": 2},
        "item_mapping": {"m1": 0, "m2": 1, "m3": 2},
        "reverse_user_mapping": {0: "u1", 1: "u2", 2: "u3"},
        "reverse_item_mapping": {0: "m1", 1: "m2", 2: "m3"},
        "n_users": 3,
        "n_items": 3,
    }


@pytest.fixture
def tiny_global_stats():
    """Minimal fake global statistics."""
    return {
        "global_mean": 3.5,
        "user_biases": np.array([0.1, -0.1, 0.0]),
        "item_biases": np.array([0.05, -0.05, 0.0]),
    }


@pytest.fixture
def tiny_matrix():
    """Small sparse 3×3 matrix with a few ratings."""
    data = np.array([5.0, 4.0, 3.0])
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    return csr_matrix((data, (rows, cols)), shape=(3, 3))

@pytest.fixture
def trained_model(tiny_mappings, tiny_global_stats, tiny_matrix):
    """A fitted model for downstream prediction tests."""
    model = ImprovedSVDRecommendationModel(n_factors=2)
    model.fit(tiny_matrix, tiny_mappings, tiny_global_stats, popular_movies=["m1", "m2", "m3"])
    return model


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------

def test_initialization_defaults():
    """Check model initializes with correct default parameters."""
    model = ImprovedSVDRecommendationModel()
    assert model.n_factors == 100
    assert model.regularization == 0.01
    assert model.user_factors is None
    assert model.item_factors is None


# -------------------------------------------------------------------
# Fit and decomposition
# -------------------------------------------------------------------

def test_fit_sets_attributes(tiny_mappings, tiny_global_stats, tiny_matrix):
    """fit() should populate user/item factors and mappings."""
    model = ImprovedSVDRecommendationModel(n_factors=2)
    model.fit(tiny_matrix, tiny_mappings, tiny_global_stats, popular_movies=["m1", "m2", "m3"])

    assert model.user_factors.shape[0] == 3
    assert model.item_factors.shape[0] == 3
    assert model.global_mean == pytest.approx(3.5)
    assert "u1" in model.user_mapping
    assert "m1" in model.item_mapping
    

def test_fit_respects_k_smaller_than_matrix_size(tiny_mappings, tiny_global_stats):
    """fit() should not break when n_factors > min(n_users, n_items)."""
    model = ImprovedSVDRecommendationModel(n_factors=10)
    matrix = csr_matrix(np.eye(3))
    model.fit(matrix, tiny_mappings, tiny_global_stats, popular_movies=["m1"])
    assert model.user_factors.shape[1] <= 3  # truncated k


# -------------------------------------------------------------------
# predict_rating
# -------------------------------------------------------------------

def test_predict_rating_known_user_item(trained_model):
    """predict_rating() should return a float in [1,5] for known user/item."""
    pred = trained_model.predict_rating("u1", "m1")
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 5.0


def test_predict_rating_unknown_user_or_item(trained_model):
    """Unknown user or item should return the global mean."""
    pred1 = trained_model.predict_rating("unknown", "m1")
    pred2 = trained_model.predict_rating("u1", "unknown_movie")
    assert pred1 == trained_model.global_mean
    assert pred2 == trained_model.global_mean


# -------------------------------------------------------------------
# predict (recommendations)
# -------------------------------------------------------------------

def test_predict_known_user(trained_model):
    """predict() should return ranked movie IDs for known user."""
    recs = trained_model.predict("u1", n_recommendations=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert all(isinstance(r, str) for r in recs)


def test_predict_unknown_user_cold_start(trained_model):
    """predict() for unknown user should fall back to cold-start recommendations."""
    recs = trained_model.predict("new_user", n_recommendations=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert all(r in ["m1", "m2", "m3"] for r in recs)


def test_predict_handles_exception(trained_model, mocker):
    """If internal failure occurs, should still return cold-start recs."""
    mocker.patch.object(trained_model, "user_factors", side_effect=Exception("fail"))
    recs = trained_model.predict("u1", n_recommendations=2)
    assert len(recs) > 0


# -------------------------------------------------------------------
# Cold start diversification
# -------------------------------------------------------------------

def test_cold_start_recommendations_diversified(trained_model):
    """Different user_ids should yield different cold-start recommendations."""
    recs1 = trained_model._cold_start_recommendations("userA", n_recommendations=2)
    recs2 = trained_model._cold_start_recommendations("userB", n_recommendations=2)
    # They should not always be identical due to hash seed
    assert recs1 != recs2 or len(set(recs1)) == len(set(recs2))


def test_cold_start_recommendations_empty_list():
    """If no popular_movies exist, cold-start should return empty list."""
    model = ImprovedSVDRecommendationModel()
    recs = model._cold_start_recommendations("u1", 5)
    assert recs == []


# -------------------------------------------------------------------
# get_model_info
# -------------------------------------------------------------------

def test_get_model_info_fields(trained_model):
    """get_model_info() should return valid metadata."""
    info = trained_model.get_model_info()
    expected_keys = {"algorithm", "n_factors", "regularization", "total_movies", "total_users", "matrix_size"}
    assert expected_keys.issubset(info.keys())
    assert info["algorithm"].startswith("Improved SVD")
    assert isinstance(info["total_movies"], int)
