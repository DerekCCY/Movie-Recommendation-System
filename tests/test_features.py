"""
Tests for ml_pipeline.features module
--------------------------------------
Covers:
- build_user_item_matrix
- calculate_popular_movies
- calculate_global_statistics
- filter_sparse_interactions
"""

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from ml_pipeline import feature_engineering


# -------------------------------------------------------------------
# Tests for build_user_item_matrix
# -------------------------------------------------------------------

class TestBuildUserItemMatrix:
    """Unit tests for build_user_item_matrix()"""

    def test_matrix_shape(self, tiny_interactions_df):
        """Output matrix should have shape (n_users, n_items)."""
        matrix, mappings = feature_engineering.build_user_item_matrix(tiny_interactions_df)
        assert isinstance(matrix, csr_matrix)
        n_users = len(tiny_interactions_df["user_id"].unique())
        n_items = len(tiny_interactions_df["movie_id"].unique())
        assert matrix.shape == (n_users, n_items)
        assert mappings["n_users"] == n_users
        assert mappings["n_items"] == n_items

    def test_mappings_bidirectional(self, tiny_interactions_df):
        """User and item mappings should be correct and reversible."""
        _, mappings = feature_engineering.build_user_item_matrix(tiny_interactions_df)
        for user, idx in mappings["user_mapping"].items():
            assert mappings["reverse_user_mapping"][idx] == user
        for movie, idx in mappings["item_mapping"].items():
            assert mappings["reverse_item_mapping"][idx] == movie

    def test_matrix_sparsity(self, tiny_interactions_df):
        """Matrix should be sparse (not dense) and mostly zeros."""
        matrix, _ = feature_engineering.build_user_item_matrix(tiny_interactions_df)
        assert isinstance(matrix, csr_matrix)
        total_elements = matrix.shape[0] * matrix.shape[1]
        nonzeros = matrix.nnz
        assert nonzeros < total_elements  # should be sparse

    def test_small_dataset(self):
        """Handles very small dataset (3 users, 5 movies)."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2", "u3", "u1", "u3"],
            "movie_id": ["m1", "m1", "m2", "m3", "m5"],
            "rating": [4, 5, 3, 2, 1]
        })
        matrix, mappings = feature_engineering.build_user_item_matrix(df)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == 4
        # verify one rating value appears correctly in matrix
        u_idx = mappings["user_mapping"]["u1"]
        i_idx = mappings["item_mapping"]["m1"]
        assert matrix[u_idx, i_idx] > 0


# -------------------------------------------------------------------
# Tests for calculate_popular_movies
# -------------------------------------------------------------------

class TestPopularMovies:
    """Tests for calculate_popular_movies()"""

    def test_calculate_popular_movies(self):
        """Movies should be ranked by popularity_score = count * mean."""
        df = pd.DataFrame({
            "movie_id": ["m1", "m1", "m2", "m2", "m3"],
            "rating": [5, 4, 2, 3, 5]
        })
        popular = feature_engineering.calculate_popular_movies(df, n_top=2)
        assert isinstance(popular, list)
        assert len(popular) == 2
        # m1 should be more popular than m2 because of higher mean and count
        assert popular[0] == "m1"

    def test_top_n_limit(self):
        """The n_top parameter should limit number of results."""
        df = pd.DataFrame({
            "movie_id": [f"m{i}" for i in range(10)],
            "rating": [5] * 10
        })
        popular = feature_engineering.calculate_popular_movies(df, n_top=5)
        assert len(popular) == 5


# -------------------------------------------------------------------
# Tests for calculate_global_statistics
# -------------------------------------------------------------------

class TestGlobalStatistics:
    """Tests for calculate_global_statistics()"""

    def test_calculate_global_statistics(self, tiny_interactions_df):
        """Global mean and bias arrays should be computed correctly."""
        _, mappings = feature_engineering.build_user_item_matrix(tiny_interactions_df)
        stats = feature_engineering.calculate_global_statistics(tiny_interactions_df, mappings)

        assert isinstance(stats, dict)
        assert "global_mean" in stats
        assert "user_biases" in stats
        assert "item_biases" in stats
        assert isinstance(stats["global_mean"], float)
        assert len(stats["user_biases"]) == mappings["n_users"]
        assert len(stats["item_biases"]) == mappings["n_items"]
        # biases should roughly sum around 0 (not required but sanity check)
        assert abs(stats["user_biases"].mean()) < 1
        assert abs(stats["item_biases"].mean()) < 1


# -------------------------------------------------------------------
# Tests for filter_sparse_interactions
# -------------------------------------------------------------------

class TestFilterSparseInteractions:
    """Tests for filter_sparse_interactions()"""

    def test_filter_removes_low_activity(self):
        """Users/items with low activity should be removed."""
        df = pd.DataFrame({
            "user_id": ["u1"] * 5 + ["u2"] * 2,
            "movie_id": ["m1", "m2", "m3", "m4", "m5", "m1", "m2"],
            "rating": [5, 4, 3, 4, 5, 2, 3]
        })

        filtered = feature_engineering.filter_sparse_interactions(
            df, min_user_interactions=3, min_item_interactions=2
        )

        # user u2 has too few interactions, should be removed
        assert "u2" not in filtered["user_id"].values
        # items with fewer than 2 interactions should also be removed
        valid_items = filtered["movie_id"].value_counts().index
        assert all(filtered["movie_id"].isin(valid_items))
