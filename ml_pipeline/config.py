"""
Configuration file for ML Pipeline

Contains all hyperparameters, paths, and constants used across the pipeline.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "cache"

# Data paths
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
INTERACTIONS_PATH = SILVER_DIR / "interactions.parquet"

# Model paths
DEFAULT_MODEL_PATH = MODELS_DIR / "improved_hybrid_model.pkl"
API_CACHE_PATH = CACHE_DIR / "movie_api_cache.json"

# A/B Testing Model Paths
MODEL_A_PATH = MODELS_DIR / os.getenv("MODEL_A_FILE", "recsys-20251113-2122/improved_hybrid_model.pkl")
MODEL_B_PATH = MODELS_DIR / os.getenv("MODEL_B_FILE", "recsys-20251116-0220/improved_hybrid_model.pkl")

# ============================================================================
# KAFKA CONFIGURATION
# ============================================================================

KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "movielog2"),
    "group_id": os.getenv("KAFKA_GROUP_ID", "mlip_ingestor"),
    "auto_offset_reset": os.getenv("AUTO_OFFSET_RESET", "earliest"),
    "enable_auto_commit": False,
}

# Batch size for Kafka consumer
KAFKA_BATCH_SIZE = int(os.getenv("FLUSH_EVERY", "5000"))

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", "http://128.2.220.241:8080"),
    "request_delay": 0.1,  # Delay between API requests (rate limiting)
    "timeout": 10,  # Request timeout in seconds
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

SVD_CONFIG = {
    "n_factors": 100,  # Number of latent factors
    "regularization": 0.01,  # Regularization parameter
    "n_popular_movies": 50,  # Number of popular movies for cold start
}

HYBRID_CONFIG = {
    "n_factors": 100,  # Number of latent factors
    "learning_rate": 0.015,  # Initial learning rate
    "regularization": 0.005,  # Regularization parameter
    "content_weight": 0.25,  # Weight for content-based features
    "n_epochs": 50,  # Number of training epochs
    "min_interactions": 3,  # Minimum interactions for users/movies
}

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

PREPROCESSING_CONFIG = {
    "min_rating": 1,  # Minimum valid rating
    "max_rating": 5,  # Maximum valid rating
    "min_user_interactions": 2,  # Filter users with fewer interactions
    "min_movie_interactions": 2,  # Filter movies with fewer interactions
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    "test_size": 0.2,  # Train/test split ratio
    "random_state": 42,  # Random seed for reproducibility
    "log_to_wandb": False,  # Whether to log to Weights & Biases
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    "k_values": [5, 10, 20],  # K values for precision@k, recall@k
    "n_inference_samples": 100,  # Number of samples for inference time measurement
}

# ============================================================================
# SERVING CONFIGURATION
# ============================================================================

SERVING_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8082)),
    "debug": False,
    "max_recommendations": 20,
    "timeout_ms": 600,  # Maximum response time in milliseconds
}
