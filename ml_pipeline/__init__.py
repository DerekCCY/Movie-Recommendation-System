"""
ML Pipeline Package for Movie Recommendation System

This package contains modular components for:
- Data ingestion (Kafka, parquet, APIs)
- Preprocessing (cleaning, deduplication, feature engineering)
- Model training (SVD collaborative filtering)
- Model evaluation (RMSE, precision@k, recall@k, etc.)
- Model serialization (save/load)
- Serving (Flask API)
"""

__version__ = "2.0.0"
