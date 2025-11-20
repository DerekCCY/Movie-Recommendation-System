"""
Movie Recommendation API Service

Flask web service for serving movie recommendations.
Endpoint: GET /recommend/<user_id>
Returns: Comma-separated list of up to 20 movie IDs

Enhanced with:
- Request validation
- Malformed request handling
- Comprehensive logging
- Provenance tracking (request IDs, model version, prediction logs)
"""

import os
import re
import time
import json
import uuid
import logging
from functools import wraps
from flask import Flask, jsonify, request, make_response
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

from ..model import ImprovedSVDRecommendationModel
from ..serialize import load_model
from .. import config

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Separate logger for malformed requests
malformed_logger = logging.getLogger('malformed_requests')
malformed_handler = logging.FileHandler('malformed_requests.log')
malformed_handler.setFormatter(logging.Formatter(
    '%(asctime)s - MALFORMED - %(message)s'
))
malformed_logger.addHandler(malformed_handler)
malformed_logger.setLevel(logging.WARNING)

app = Flask(__name__)

# ---------- Provenance logging setup ----------
PRED_LOG_DIR = Path(os.getenv("PRED_LOG_DIR", "logs/predictions"))
PRED_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _write_prediction_log(record: dict) -> None:
    """
    Write prediction record to daily JSONL file for provenance tracking.

    Args:
        record: Dictionary containing prediction details (request_id, user_id, recommendations, etc.)
    """
    day = time.strftime("%Y-%m-%d")
    path = PRED_LOG_DIR / f"{day}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _infer_model_version_and_card(model_path: str) -> Tuple[str, dict]:
    """
    Try to infer version directory from model_path and load its model_card.json if present.

    Args:
        model_path: Path to model file

    Returns:
        Tuple of (model_version, model_card_dict). Falls back to ("unknown", {}).
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
        # Check if it's production model
        if "production" in str(p):
            # Read production version file if exists
            version_file = p.parent / "current_version.txt"
            if version_file.exists():
                version = version_file.read_text().strip()
                # Try to find model card in the versioned directory
                versioned_dir = p.parent.parent / version
                if versioned_dir.exists():
                    card_path = versioned_dir / "model_card.json"
                    if card_path.exists():
                        with open(card_path) as f:
                            return version, json.load(f)
                return version, {}
    except Exception as e:
        logger.warning(f"Failed to infer model version/card: {e}")
    return "unknown", {}

# -----------------------------------------------


class RequestValidator:
    """
    Validates API requests for correct format and content.
    """
    
    # Validation rules
    MAX_USER_ID = 1_000_000  # 1 million users
    MIN_RECOMMENDATIONS = 1
    MAX_RECOMMENDATIONS = 100
    
    @staticmethod
    def validate_user_id(user_id: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate user_id format - must be a positive integer.
        
        Args:
            user_id: User identifier to validate (as string from URL)
            
        Returns:
            Tuple of (is_valid, error_message, parsed_user_id)
        """
        if not user_id:
            return False, "user_id is required and cannot be empty", None
        
        if not isinstance(user_id, str):
            return False, f"user_id must be provided as string, got {type(user_id).__name__}", None
        
        # Try to parse as integer
        try:
            user_id_int = int(user_id)
        except ValueError:
            return False, f"user_id must be a positive integer, got '{user_id}'", None
        
        # Check if positive
        if user_id_int <= 0:
            return False, f"user_id must be a positive integer (greater than 0), got {user_id_int}", None
        
        return True, None, user_id_int
    
    @staticmethod
    def validate_n_recommendations(n: any) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate n_recommendations parameter - must be positive integer up to 100.
        
        Args:
            n: Number of recommendations parameter
            
        Returns:
            Tuple of (is_valid, error_message, parsed_value)
        """
        if n is None:
            return True, None, 20  # Default value
        
        try:
            n_int = int(n)
        except (ValueError, TypeError):
            return False, f"n_recommendations must be a positive integer, got '{n}'", None
        
        if n_int < RequestValidator.MIN_RECOMMENDATIONS:
            return False, f"n_recommendations must be a positive integer (at least {RequestValidator.MIN_RECOMMENDATIONS}), got {n_int}", None
        
        if n_int > RequestValidator.MAX_RECOMMENDATIONS:
            return False, f"n_recommendations cannot exceed {RequestValidator.MAX_RECOMMENDATIONS}, got {n_int}", None
        
        return True, None, n_int


def log_malformed_request(error_type: str, details: Dict) -> None:
    """
    Log malformed request with detailed information.
    
    Args:
        error_type: Type of validation error
        details: Dictionary with request details
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'error_type': error_type,
        'ip': request.remote_addr,
        'endpoint': request.endpoint,
        'path': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'details': details
    }
    
    malformed_logger.warning(f"{error_type}: {log_entry}")


def validate_request(f):
    """
    Decorator to validate incoming requests.
    Checks request format, parameters, and logs malformed requests.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract user_id from kwargs (comes from URL path)
        user_id = kwargs.get('user_id')
        
        # Validate user_id
        is_valid, error_msg, parsed_user_id = RequestValidator.validate_user_id(user_id)
        if not is_valid:
            log_malformed_request(
                'INVALID_USER_ID',
                {
                    'user_id': user_id,
                    'error': error_msg,
                    'query_params': dict(request.args)
                }
            )
            return jsonify({
                'error': 'Invalid request',
                'message': error_msg,
                'user_id': user_id
            }), 400
        
        # Use parsed integer user_id
        kwargs['user_id'] = str(parsed_user_id)  # Convert back to string for model compatibility
        
        # Validate query parameters
        n_recommendations = request.args.get('n', None)
        is_valid, error_msg, parsed_n = RequestValidator.validate_n_recommendations(n_recommendations)
        
        if not is_valid:
            log_malformed_request(
                'INVALID_PARAMETER',
                {
                    'user_id': user_id,
                    'parameter': 'n_recommendations',
                    'value': n_recommendations,
                    'error': error_msg
                }
            )
            return jsonify({
                'error': 'Invalid request',
                'message': error_msg,
                'parameter': 'n_recommendations'
            }), 400
        
        # Add validated parameter to kwargs
        kwargs['n_recommendations'] = parsed_n
        
        # Check for unexpected query parameters
        allowed_params = {'n'}
        unexpected_params = set(request.args.keys()) - allowed_params
        if unexpected_params:
            log_malformed_request(
                'UNEXPECTED_PARAMETERS',
                {
                    'user_id': user_id,
                    'unexpected_params': list(unexpected_params),
                    'all_params': dict(request.args)
                }
            )
            # Don't fail, just log warning
            logger.warning(f"Unexpected parameters in request: {unexpected_params}")
        
        return f(*args, **kwargs)
    
    return decorated_function


class RecommenderService:
    """
    Wrapper service for the trained SVD recommendation model.
    Handles model loading and recommendation generation.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize recommender service.

        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.model_path = model_path

        # Provenance metadata
        self.model_version = "unknown"
        self.model_card = {}
        self.pipeline_git_commit = "unknown"

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to pickled model file
        """
        try:
            self.model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")

            # Load provenance metadata
            self.model_version, self.model_card = _infer_model_version_and_card(model_path)
            self.pipeline_git_commit = self.model_card.get("training_commit", "unknown")

            logger.info(f"Model version for provenance: {self.model_version}")
            logger.info(f"Training commit (from card): {self.pipeline_git_commit}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def get_recommendations(self, user_id: str, n_recommendations: int = 20) -> Tuple[List[str], str]:
        """
        Get movie recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return

        Returns:
            Tuple of (List of movie IDs, request_id for provenance tracking)

        Note:
            - For known users: returns personalized SVD recommendations
            - For unknown users: returns diversified popular movies (cold start)
        """
        start_time = time.time()

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        recommendations = self.model.predict(user_id, n_recommendations=n_recommendations)

        response_time = time.time() - start_time

        # Generate request ID and write prediction provenance log
        req_id = str(uuid.uuid4())
        record = {
            "request_id": req_id,
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",  # ISO 8601 with milliseconds
            "user_id": str(user_id),
            "topk": [{"item_id": str(m), "score": None} for m in recommendations],
            "latency_ms": int(response_time * 1000),
            "model_version": self.model_version,
            "pipeline_git_commit": self.pipeline_git_commit,
        }
        _write_prediction_log(record)

        return recommendations, req_id


# Global service instance
recommender = None


def initialize_service(model_path: str = None) -> None:
    """
    Initialize the recommender service on startup.

    Args:
        model_path: Path to model file (uses config default if None)
    """
    global recommender

    # Use config default if model_path is None
    if model_path is None:
        model_path = os.getenv('MODEL_PATH', config.DEFAULT_MODEL_PATH)

    recommender = RecommenderService(model_path)
    logger.info("Recommender service initialized")


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request was malformed or invalid'
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    log_malformed_request(
        'ENDPOINT_NOT_FOUND',
        {
            'path': request.path,
            'method': request.method
        }
    )
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/health', '/recommend/<user_id>', '/']
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 Method Not Allowed errors."""
    log_malformed_request(
        'METHOD_NOT_ALLOWED',
        {
            'path': request.path,
            'method': request.method,
            'allowed_methods': error.valid_methods if hasattr(error, 'valid_methods') else []
        }
    )
    return jsonify({
        'error': 'Method Not Allowed',
        'message': f'The {request.method} method is not allowed for this endpoint',
        'allowed_methods': list(error.valid_methods) if hasattr(error, 'valid_methods') else []
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


@app.before_request
def log_request():
    """Log all incoming requests."""
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON with service status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender.model is not None if recommender else False,
        'model_version': getattr(recommender, "model_version", "unknown") if recommender else "unknown",
        'timestamp': datetime.now().isoformat()
    })


@app.route('/recommend/<user_id>', methods=['GET'])
@validate_request
def recommend_movies(user_id, n_recommendations=20):
    """
    Get movie recommendations for a user.

    Args:
        user_id: User identifier from URL path (validated)

    Query Parameters:
        n: Number of recommendations to return (1-100, default: 20)

    Returns:
        Plain text comma-separated list of movie IDs

    Example:
        GET /recommend/12345
        Response: movie_1,movie_2,movie_3,...,movie_20

        GET /recommend/12345?n=10
        Response: movie_1,movie_2,...,movie_10

    Error Responses:
        400: Invalid user_id or parameters
        500: Server error
    """
    start_time = time.time()

    try:
        if not recommender:
            logger.warning("Recommender not initialized, initializing now")
            initialize_service()

        # Get recommendations (returns tuple with request_id for provenance)
        recommendations, req_id = recommender.get_recommendations(user_id, n_recommendations=n_recommendations)

        # Format as comma-separated string
        result = ','.join(map(str, recommendations))

        response_time = time.time() - start_time

        # Log successful request
        logger.info(
            f"SUCCESS - user={user_id}, n_requested={n_recommendations}, "
            f"n_returned={len(recommendations)}, response_time={response_time:.3f}s"
        )

        # Warn if response is slow (>600ms requirement)
        if response_time > 0.6:
            logger.warning(f"SLOW_RESPONSE: {response_time:.3f}s for user {user_id}")

        # Create response with provenance headers
        resp = make_response(result, 200)
        resp.headers['Content-Type'] = 'text/plain'
        resp.headers['X-Request-Id'] = req_id
        resp.headers['X-Model-Version'] = getattr(recommender, "model_version", "unknown")
        return resp

    except ValueError as e:
        # Handle specific validation errors from model
        logger.error(f"ValueError in recommend endpoint: {e}", exc_info=True)
        log_malformed_request(
            'MODEL_VALIDATION_ERROR',
            {
                'user_id': user_id,
                'error': str(e)
            }
        )
        return jsonify({
            'error': 'Invalid request',
            'message': str(e)
        }), 400

    except RuntimeError as e:
        # Handle model not loaded errors
        logger.error(f"RuntimeError in recommend endpoint: {e}")
        return jsonify({
            'error': 'Service error',
            'message': 'Model not available'
        }), 503

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in recommend endpoint: {e}", exc_info=True)

        # Generate fallback request ID for error tracking
        req_id = str(uuid.uuid4())
        resp = make_response(jsonify({
            'error': 'Internal error',
            'message': 'Error generating recommendations'
        }), 500)
        resp.headers['X-Request-Id'] = req_id
        resp.headers['X-Model-Version'] = getattr(recommender, "model_version", "unknown") if recommender else "unknown"
        return resp


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with service information.

    Returns:
        JSON with service metadata
    """
    return jsonify({
        'service': 'Movie Recommendation API',
        'version': '2.0',
        'endpoints': {
            'health': {
                'path': '/health',
                'method': 'GET',
                'description': 'Check service health'
            },
            'recommend': {
                'path': '/recommend/<user_id>',
                'method': 'GET',
                'description': 'Get movie recommendations',
                'parameters': {
                    'user_id': 'Required path parameter (positive integer)',
                    'n': 'Optional query parameter (positive integer, 1-100, default: 20)'
                },
                'example': '/recommend/12345?n=10'
            }
        },
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Initialize service on startup
    # Get model_path and port from config
    model_path = os.getenv('MODEL_PATH', config.DEFAULT_MODEL_PATH)
    
    try:
        initialize_service(model_path)
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        logger.warning("Service starting without model - recommendations will fail")

    # Run Flask app
    host = config.SERVING_CONFIG['host']
    port = config.SERVING_CONFIG['port']
    debug = config.SERVING_CONFIG['debug']

    logger.info(f"Starting Flask server on {host}:{port}")
    logger.info(f"Logging to: api_requests.log and malformed_requests.log")
    
    app.run(host=host, port=port, debug=debug)