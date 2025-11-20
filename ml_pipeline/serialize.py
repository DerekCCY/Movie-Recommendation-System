"""
Model Serialization Module

Handles saving and loading trained models to/from disk.
"""

import pickle
import os
from typing import Any
from pathlib import Path


def save_model(model: Any, path: str) -> None:
    """
    Save trained model to pickle file.

    Args:
        model: Trained model object (ImprovedSVDRecommendationModel)
        path: File path to save model (.pkl extension)

    Example:
        >>> save_model(trained_model, "models/svd_model.pkl")
    """
    # Ensure directory exists
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {path}")


def load_model(path: str) -> Any:
    """
    Load trained model from pickle file.

    Args:
        path: File path to saved model (.pkl)

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist

    Example:
        >>> model = load_model("models/svd_model.pkl")
        >>> recommendations = model.predict("user_123")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded from: {path}")
    return model


def get_model_size(path: str) -> float:
    """
    Get size of saved model file in megabytes.

    Metric: Disk space required for model
    Data: Serialized model file
    Operationalization: File size in bytes / (1024^2)

    Args:
        path: Path to model file

    Returns:
        Model size in MB

    Example:
        >>> size_mb = get_model_size("models/svd_model.pkl")
        >>> print(f"Model size: {size_mb:.2f} MB")
        Model size: 12.35 MB
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 ** 2)

    return size_mb


def verify_model_integrity(path: str) -> bool:
    """
    Verify that a saved model can be loaded successfully.

    Args:
        path: Path to model file

    Returns:
        True if model loads successfully, False otherwise
    """
    try:
        load_model(path)
        return True
    except Exception as e:
        print(f"Model integrity check failed: {e}")
        return False
