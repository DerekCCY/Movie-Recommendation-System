"""
Movie Recommendation API Service
Serves SVD-based movie recommendations on port 8082
"""

import os
import time
import json                              # ### ADDED
import uuid                              # ### ADDED
import logging
from flask import Flask, jsonify, make_response   # ### CHANGED: import make_response
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import from ml_pipeline package
import sys
from pathlib import Path

# Add parent directory to path to import ml_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_pipeline.model import ImprovedSVDRecommendationModel
from ml_pipeline.serialize import load_model as ml_load_model
from ml_pipeline import config as ml_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------- Provenance helpers (self-contained) ----------  # ### ADDED
PRED_LOG_DIR = Path(os.getenv("PRED_LOG_DIR", "logs/predictions"))
PRED_LOG_DIR.mkdir(parents=True, exist_ok=True)
#IMAGE_DIGEST = os.getenv("IMAGE_DIGEST")  # optional, just nice-to-have

def _write_prediction_log(record: dict) -> None:
    day = time.strftime("%Y-%m-%d")
    path = PRED_LOG_DIR / f"{day}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _infer_model_version_and_card(model_path: str) -> tuple[str, dict]:
    """
    Try to infer version directory from model_path and load its model_card.json if present.
    Returns (model_version, model_card_dict). Falls back to ("unknown", {}).
    """
    try:
        p = Path(model_path).resolve()
        # Common layout: models/recsys-YYYYMMDD-HHMM/<model_file>
        if p.parent.name.startswith("recsys-"):
            version = p.parent.name
            card_path = p.parent / "model_card.json"
            if card_path.exists():
                with open(card_path) as f:
                    return version, json.load(f)
            return version, {}
        # Fallback: search upward for a recsys-* folder
        for parent in p.parents:
            if parent.name.startswith("recsys-"):
                card_path = parent / "model_card.json"
                if card_path.exists():
                    with open(card_path) as f:
                        return parent.name, json.load(f)
                return parent.name, {}
    except Exception as e:
        logger.warning(f"Failed to infer model version/card: {e}")
    return "unknown", {}

# ----------------------------------------------------------

class SVDRecommenderWrapper:
    """Wrapper for the trained ImprovedSVDRecommendationModel"""

    def __init__(self, model_path=None, interactions_path=None):
        self.model = None
        self.interactions_df = None
        self.popular_movies = []

        # ### ADDED: metadata for provenance
        self.model_version = "unknown"
        self.model_card = {}
        self.pipeline_git_commit = "unknown"

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        if interactions_path and os.path.exists(interactions_path):
            self.load_interactions(interactions_path)

    def load_model(self, model_path):
        """Load trained SVD model using ml_pipeline.serialize"""
        try:
            # Use ml_pipeline's load_model function
            self.model = ml_load_model(str(model_path))

            # Load provenance metadata
            self.model_version, self.model_card = _infer_model_version_and_card(model_path)
            self.pipeline_git_commit = self.model_card.get("training_commit", "unknown")

            logger.info(f"Model loaded from {model_path} via ml_pipeline.serialize")
            logger.info(f"Model version for provenance: {self.model_version}")
            logger.info(f"Training commit (from card): {self.pipeline_git_commit}")

            logger.info(f"Model has {len(self.model.user_ids)} users and {len(self.model.all_movie_ids)} movies")

            # Use the model's built-in popular movies
            if hasattr(self.model, 'popular_movies') and self.model.popular_movies:
                self.popular_movies = self.model.popular_movies
                logger.info(f"Using {len(self.popular_movies)} popular movies for cold start")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def load_interactions(self, interactions_path):
        """Load interaction data for additional cold start handling"""
        try:
            self.interactions_df = pd.read_parquet(interactions_path)
            logger.info(f"Interactions loaded: {len(self.interactions_df)} records")

            # If model doesn't have popular movies, create them from interactions
            if not self.popular_movies and self.interactions_df is not None:
                # Get popular movies based on explicit ratings
                explicit_ratings = self.interactions_df[self.interactions_df['rating'].notna()]
                if len(explicit_ratings) > 0:
                    movie_popularity = explicit_ratings.groupby('movie_id').agg({
                        'rating': ['count', 'mean']
                    })
                    movie_popularity.columns = ['rating_count', 'avg_rating']
                    movie_popularity['popularity_score'] = (
                        movie_popularity['rating_count'] * movie_popularity['avg_rating']
                    )
                    self.popular_movies = (movie_popularity
                                         .sort_values('popularity_score', ascending=False)
                                         .head(50)
                                         .index.tolist())
                    logger.info(f"Created {len(self.popular_movies)} popular movies from interactions")

        except Exception as e:
            logger.error(f"Failed to load interactions: {e}")

    def get_recommendations(self, user_id, n_recommendations=20):
        """Get movie recommendations for a user"""
        start_time = time.time()

        try:
            # Convert user_id to string to match model format
            user_id_str = str(user_id)

            # Use the model's predict method
            if self.model and hasattr(self.model, 'predict'):
                recommendations = self.model.predict(user_id_str, n_recommendations=n_recommendations)

                # If no recommendations or user not found, model will return popular movies
                if not recommendations:
                    recommendations = self.popular_movies[:n_recommendations]
                    logger.info(f"Using fallback popular movies for user {user_id}")
                else:
                    logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")

            else:
                # Fallback to popular movies if model doesn't work
                recommendations = self.popular_movies[:n_recommendations]
                logger.warning(f"Model prediction failed, using popular movies for user {user_id}")

            response_time = time.time() - start_time
            logger.info(f"Recommendations generated for user {user_id} in {response_time:.3f}s")

            # === ADDED: write prediction provenance log ===
            req_id = str(uuid.uuid4())
            record = {
                "request_id": req_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                "user_id": str(user_id),
                "topk": [{"item_id": str(m), "score": None} for m in recommendations],
                "latency_ms": int(response_time * 1000),
                "model_version": self.model_version,
                "pipeline_git_commit": self.pipeline_git_commit,
                #"serving_image": IMAGE_DIGEST,
                # You can add experiment headers later in the route if needed
            }
            _write_prediction_log(record)
            # return recs + the req_id so caller can trace (we'll add header outside)
            return recommendations[:n_recommendations], req_id

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Final fallback to popular movies
            return self.popular_movies[:n_recommendations], str(uuid.uuid4())

# Global recommender instance
recommender = None

def initialize_recommender():
    """Initialize the recommender with model and data"""
    global recommender

    # Use ml_pipeline.config defaults
    model_path = os.getenv('MODEL_PATH', str(ml_config.DEFAULT_MODEL_PATH))
    interactions_path = os.getenv('INTERACTIONS_PATH', str(ml_config.INTERACTIONS_PATH))

    recommender = SVDRecommenderWrapper(model_path, interactions_path)
    logger.info("Recommender initialized")
    logger.info(f"Using model path: {model_path}")
    logger.info(f"Using interactions path: {interactions_path}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender.model is not None,
        'data_loaded': recommender.interactions_df is not None,
        'model_version': getattr(recommender, "model_version", "unknown")  # ### ADDED
    })

@app.route('/recommend/<user_id>', methods=['GET'])
def recommend_movies(user_id):
    """
    Get movie recommendations for a user
    Returns comma-separated list of movie IDs
    """
    start_time = time.time()

    try:
        if not recommender:
            initialize_recommender()

        # Convert user_id to appropriate format
        try:
            user_id_cast = int(user_id)
        except ValueError:
            user_id_cast = str(user_id)

        # Get recommendations (+ provenance request id)
        recommendations, req_id = recommender.get_recommendations(user_id_cast, n_recommendations=20)

        # Format as comma-separated string
        result = ','.join(map(str, recommendations))

        response_time = time.time() - start_time

        # Log for monitoring
        logger.info(f"Request: user={user_id}, response_time={response_time:.3f}s, movies={len(recommendations)}")

        # Ensure response time is under 600ms
        if response_time > 0.6:
            logger.warning(f"Slow response: {response_time:.3f}s for user {user_id}")

        # === ADDED: include request_id in response header for easy tracing ===
        resp = make_response(result, 200)
        resp.headers['Content-Type'] = 'text/plain'
        resp.headers['X-Request-Id'] = req_id
        resp.headers['X-Model-Version'] = getattr(recommender, "model_version", "unknown")
        return resp

    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        # Return popular movies as fallback
        fallback = ','.join(map(str, recommender.popular_movies[:20]))
        resp = make_response(fallback, 200)
        resp.headers['Content-Type'] = 'text/plain'
        resp.headers['X-Request-Id'] = str(uuid.uuid4())
        resp.headers['X-Model-Version'] = getattr(recommender, "model_version", "unknown")
        return resp

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Movie Recommendation API',
        'team': 'team02',
        'endpoints': {
            'health': '/health',
            'recommend': '/recommend/<user_id>'
        }
    })

# Initialize recommender on startup
initialize_recommender()

if __name__ == '__main__':
    # Run on port 8082 as required
    port = int(os.getenv('PORT', 8082))
    app.run(host='0.0.0.0', port=port, debug=False)
