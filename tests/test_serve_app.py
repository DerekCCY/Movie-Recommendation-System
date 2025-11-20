"""
Tests for ml_pipeline.serve.app (Flask API)

Covers:
- /health
- /recommend/<user_id>
- Root endpoint
- Error and timing handling
"""

import pytest
import time
from ml_pipeline.serve import app


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def flask_client():
    """Provide a Flask test client."""
    test_app = app.app
    test_app.config.update({"TESTING": True})
    return test_app.test_client()


@pytest.fixture
def mock_recommender(mocker):
    """Mock recommender with deterministic output."""
    mock = mocker.MagicMock()
    mock.get_recommendations.return_value = (["movie_1", "movie_2", "movie_3"], "test-request-id")
    mock.model_version = "test-version"
    return mock


# -------------------------------------------------------------------
# Endpoint tests
# -------------------------------------------------------------------

class TestFlaskEndpoints:
    """Integration tests for Flask endpoints."""

    def test_health_endpoint(self, flask_client, mocker):
        """GET /health returns JSON with correct fields."""
        mock_recommender = mocker.MagicMock()
        mock_recommender.model = object()
        mock_recommender.model_version = "test-version"
        mocker.patch("ml_pipeline.serve.app.recommender", mock_recommender)

        response = flask_client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_recommend_endpoint_known_user(self, flask_client, mocker, mock_recommender):
        """Valid numeric user_id returns plain-text recommendations."""
        mocker.patch("ml_pipeline.serve.app.recommender", mock_recommender)

        response = flask_client.get("/recommend/123")
        assert response.status_code == 200
        assert response.mimetype == "text/plain"

        text = response.data.decode()
        assert "movie_1" in text
        assert "," in text

    def test_recommend_endpoint_invalid_user_id(self, flask_client):
        """Non-numeric user_id should return 400 JSON error."""
        response = flask_client.get("/recommend/user_abc")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "Invalid request" in data["error"] or "Bad Request" in data["error"]

    def test_recommend_endpoint_unknown_user(self, flask_client, mocker):
        """Numeric unknown user still returns 200 and fallback recommendations."""
        mock_recommender = mocker.MagicMock()
        mock_recommender.get_recommendations.return_value = (["popular_1", "popular_2"], "test-request-id")
        mock_recommender.model_version = "test-version"
        mocker.patch("ml_pipeline.serve.app.recommender", mock_recommender)

        response = flask_client.get("/recommend/456")
        assert response.status_code == 200
        text = response.data.decode()
        assert "popular_1" in text

    def test_recommend_response_format(self, flask_client, mocker, mock_recommender):
        """Ensure response is text/plain (comma-separated)."""
        mocker.patch("ml_pipeline.serve.app.recommender", mock_recommender)
        response = flask_client.get("/recommend/789")
        assert response.mimetype == "text/plain"
        text = response.data.decode()
        assert not text.strip().startswith("{")

    def test_recommend_response_time(self, flask_client, mocker):
        """Response should complete quickly (< 600ms)."""
        mock_recommender = mocker.MagicMock()
        mock_recommender.get_recommendations.return_value = (["a", "b", "c"], "test-request-id")
        mock_recommender.model_version = "test-version"
        mocker.patch("ml_pipeline.serve.app.recommender", mock_recommender)

        start = time.time()
        response = flask_client.get("/recommend/42")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.6, f"Response too slow: {elapsed:.3f}s"

    def test_error_handling_internal_error(self, flask_client, mocker):
        """Unhandled exceptions should return 500 JSON error."""
        mock_recommender = mocker.MagicMock()
        mock_recommender.get_recommendations.side_effect = Exception("Boom!")
        mocker.patch("ml_pipeline.serve.app.recommender", mock_recommender)

        response = flask_client.get("/recommend/99")
        assert response.status_code == 500
        assert b"Error generating recommendations" in response.data


# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------

class TestRootEndpoint:
    """Basic sanity test for / endpoint."""

    def test_root_endpoint(self, flask_client):
        response = flask_client.get("/")
        assert response.status_code == 200
        data = response.get_json()
        assert "service" in data
        assert "endpoints" in data
        assert "recommend" in data["endpoints"]
