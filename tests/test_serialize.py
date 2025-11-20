"""
Tests for ml_pipeline.serialize module
---------------------------------------
Covers:
- save_model()
- load_model()
- get_model_size()
- verify_model_integrity()
"""

import pytest
import pickle
from pathlib import Path
from ml_pipeline import serialize


# -------------------------------------------------------------------
# Dummy model class for testing
# -------------------------------------------------------------------

class DummyModel:
    """Simple mock model with a predict() method."""
    def __init__(self):
        self.n_users = 5
        self.n_items = 10
        self.global_mean = 3.5

    def predict(self, user_id, n_recommendations=5):
        return [f"movie_{i}" for i in range(n_recommendations)]


# -------------------------------------------------------------------
# Tests for saving and loading models
# -------------------------------------------------------------------

class TestSaveLoadModel:
    """Tests for save_model(), load_model(), get_model_size()"""

    def test_save_model(self, tmp_path):
        """Saving a model should create a pickle file."""
        model = DummyModel()
        model_path = tmp_path / "model.pkl"

        serialize.save_model(model, model_path)
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_load_model(self, tmp_path):
        """Loading a saved model should return a Python object."""
        model = DummyModel()
        model_path = tmp_path / "model.pkl"
        serialize.save_model(model, model_path)

        loaded = serialize.load_model(model_path)
        assert isinstance(loaded, DummyModel)
        assert loaded.n_users == model.n_users
        assert hasattr(loaded, "predict")

    def test_roundtrip(self, tmp_path):
        """Saving and loading should preserve model behavior."""
        model = DummyModel()
        model_path = tmp_path / "roundtrip.pkl"
        serialize.save_model(model, model_path)

        loaded = serialize.load_model(model_path)
        # Check that predict() produces same output
        assert model.predict("u1") == loaded.predict("u1")

    def test_get_model_size(self, tmp_path):
        """File size should be a positive float in MB."""
        model = DummyModel()
        model_path = tmp_path / "size.pkl"
        serialize.save_model(model, model_path)

        size = serialize.get_model_size(model_path)
        assert isinstance(size, float)
        assert size > 0.0

    def test_load_nonexistent_file(self, tmp_path):
        """Loading a nonexistent model should raise FileNotFoundError."""
        bad_path = tmp_path / "missing.pkl"
        with pytest.raises(FileNotFoundError):
            serialize.load_model(bad_path)

    def test_get_model_size_nonexistent(self, tmp_path):
        """Getting size for nonexistent model should raise FileNotFoundError."""
        bad_path = tmp_path / "missing.pkl"
        with pytest.raises(FileNotFoundError):
            serialize.get_model_size(bad_path)


# -------------------------------------------------------------------
# Tests for verify_model_integrity()
# -------------------------------------------------------------------

class TestModelIntegrity:
    """Tests for verify_model_integrity()"""

    def test_verify_model_integrity(self, tmp_path):
        """A valid model file should return True."""
        model = DummyModel()
        model_path = tmp_path / "valid.pkl"
        serialize.save_model(model, model_path)

        assert serialize.verify_model_integrity(model_path) is True

    def test_corrupted_file_handling(self, tmp_path):
        """Corrupted pickle file should return False and not raise."""
        bad_file = tmp_path / "corrupted.pkl"
        # Write invalid bytes to file
        bad_file.write_bytes(b"not_a_pickle_file")

        result = serialize.verify_model_integrity(bad_file)
        assert result is False
