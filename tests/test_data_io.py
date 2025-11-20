"""
Tests for ml_pipeline.data_io module

Covers:
- Loading interactions from parquet
- MovieAPIClient basic behavior (mocked HTTP)
- Caching, error handling, and data quality validation
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from ml_pipeline.data_io import (
    load_interactions_from_parquet,
    MovieAPIClient,
    DataQualityValidator,
    DataQualityReport,
)

# ---------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------

class TestLoadInteractionsFromParquet:
    """Tests for load_interactions_from_parquet function"""

    def test_load_valid_parquet(self, sample_interactions_path):
        """Loading a valid parquet returns a non-empty DataFrame."""
        df = load_interactions_from_parquet(sample_interactions_path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_nonexistent_file(self, tmp_path):
        """A non-existent path should raise FileNotFoundError."""
        missing = tmp_path / "no_such_file.parquet"
        with pytest.raises(FileNotFoundError):
            _ = load_interactions_from_parquet(missing)

    def test_output_schema_minimal(self, sample_interactions_path):
        """Output should contain at least user_id and movie_id."""
        df = load_interactions_from_parquet(sample_interactions_path)
        cols = set(df.columns.str.lower())
        assert {"user_id", "movie_id"}.issubset(cols)

    def test_determinism(self, sample_interactions_path):
        """Loading the same file twice yields identical content."""
        df1 = load_interactions_from_parquet(sample_interactions_path)
        df2 = load_interactions_from_parquet(sample_interactions_path)
        pd.testing.assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_like=True)


# ---------------------------------------------------------------------
# MovieAPIClient (HTTP mocked)
# ---------------------------------------------------------------------

class TestMovieAPIClient:
    """Lightweight tests for MovieAPIClient"""

    def test_initialization(self):
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)
        assert isinstance(client, MovieAPIClient)
        # quality_reports list should exist
        assert isinstance(client.quality_reports, list)

    def test_get_movie_success_and_validation(self, mocker):
        """Successful call returns parsed dict and records validation."""
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)

        fake_resp = mocker.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "id": "movie_a",
            "title": "Test Movie",
            "genres": ["Action", "Drama"],
            "year": 2020,
        }

        get_spy = mocker.patch.object(client.session, "get", return_value=fake_resp)

        data = client.get_movie("movie_a")
        assert isinstance(data, dict)
        assert data["id"] == "movie_a"
        get_spy.assert_called_once()

        # Should have at least one quality report recorded
        assert client.quality_reports
        rep = client.quality_reports[0]
        assert isinstance(rep, DataQualityReport)
        assert rep.entity_type == "movie"

    def test_get_movie_cache_hit(self, mocker):
        """Repeated calls should use cache and skip HTTP."""
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)

        fake_resp = mocker.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"id": "m1", "title": "Movie 1"}
        get_spy = mocker.patch.object(client.session, "get", return_value=fake_resp)

        d1 = client.get_movie("m1")
        d2 = client.get_movie("m1")
        assert d1 == d2
        get_spy.assert_called_once()

    def test_get_user_success(self, mocker):
        """Happy path for get_user (if implemented)."""
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)
        fake_resp = mocker.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"id": "u1", "age": 30}
        get_spy = mocker.patch.object(client.session, "get", return_value=fake_resp)

        data = client.get_user("u1")
        assert data["id"] == "u1"
        assert client.quality_reports
        get_spy.assert_called_once()

    def test_rate_limiting_calls_sleep(self, mocker):
        """When request_delay>0, time.sleep should be called at least once."""
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0.01)

        fake_resp = mocker.Mock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"id": "m1", "title": "A"}
        mocker.patch.object(client.session, "get", return_value=fake_resp)
        sleep_spy = mocker.patch("ml_pipeline.data_io.time.sleep")

        client.get_movie("m1")
        assert sleep_spy.called

    def test_error_handling_returns_none(self, mocker):
        """HTTP error or exception should return None and not raise."""
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)

        # Case A: HTTP 404
        resp_404 = mocker.Mock()
        resp_404.status_code = 404
        mocker.patch.object(client.session, "get", return_value=resp_404)
        assert client.get_movie("bad") is None

        # Case B: Exception
        mocker.patch.object(client.session, "get", side_effect=Exception("timeout"))
        assert client.get_movie("fail") is None

    def test_cache_save_and_load_roundtrip(self, tmp_path):
        """save_cache/load_cache should persist data correctly."""
        path = tmp_path / "cache.json"
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)
        client.cache = {"m1": {"id": "m1", "title": "Movie"}}
        client.save_cache(path)
        assert path.exists()

        client2 = MovieAPIClient(base_url="https://api.example.com", request_delay=0)
        client2.load_cache(path)
        assert "m1" in client2.cache
        assert client2.cache["m1"]["title"] == "Movie"

    def test_export_quality_report_json(self, tmp_path):
        """export_quality_report should create a valid JSON summary."""
        path = tmp_path / "report.json"
        client = MovieAPIClient(base_url="https://api.example.com", request_delay=0)
        # simulate one report
        client.quality_reports = [
            DataQualityReport(
                entity_type="movie",
                entity_id="m1",
                is_valid=True,
                errors=[],
                warnings=["Missing optional field: runtime"],
                timestamp="2025-10-27T00:00:00",
            )
        ]
        client.export_quality_report(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "reports" in data
        assert data["reports"][0]["entity_id"] == "m1"
