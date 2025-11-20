"""
Movie Recommendation API Service with A/B Test Logging

Key additions:
1. Integrated ABTestLogger for experiment tracking
2. New /experiment/stats endpoint for live statistics
3. Enhanced logging of all recommendation requests
"""

import os
import time
import logging
from functools import wraps
from flask import Flask, jsonify, request
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib

# Import your existing components
from ..model import ImprovedSVDRecommendationModel
from ..serialize import load_model
from .. import config

# Import the new A/B test logger
from .ab_test_logger import ab_logger

# AB testing model paths
MODEL_A_PATH = config.MODEL_A_PATH
MODEL_B_PATH = config.MODEL_B_PATH

# A/B testing global variables
recommender_A = None
recommender_B = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

malformed_logger = logging.getLogger('malformed_requests')
malformed_handler = logging.FileHandler('malformed_requests.log')
malformed_handler.setFormatter(logging.Formatter(
    '%(asctime)s - MALFORMED - %(message)s'
))
malformed_logger.addHandler(malformed_handler)
malformed_logger.setLevel(logging.WARNING)

app = Flask(__name__)


class RequestValidator:
    """Validates API requests for correct format and content."""
    
    MAX_USER_ID = 1_000_000
    MIN_RECOMMENDATIONS = 1
    MAX_RECOMMENDATIONS = 100
    
    @staticmethod
    def validate_user_id(user_id: str) -> Tuple[bool, Optional[str], Optional[int]]:
        if not user_id:
            return False, "user_id is required and cannot be empty", None
        
        if not isinstance(user_id, str):
            return False, f"user_id must be provided as string, got {type(user_id).__name__}", None
        
        try:
            user_id_int = int(user_id)
        except ValueError:
            return False, f"user_id must be a positive integer, got '{user_id}'", None
        
        if user_id_int <= 0:
            return False, f"user_id must be a positive integer (greater than 0), got {user_id_int}", None
        
        return True, None, user_id_int
    
    @staticmethod
    def validate_n_recommendations(n: any) -> Tuple[bool, Optional[str], Optional[int]]:
        if n is None:
            return True, None, 20
        
        try:
            n_int = int(n)
        except (ValueError, TypeError):
            return False, f"n_recommendations must be a positive integer, got '{n}'", None
        
        if n_int < RequestValidator.MIN_RECOMMENDATIONS:
            return False, f"n_recommendations must be at least {RequestValidator.MIN_RECOMMENDATIONS}, got {n_int}", None
        
        if n_int > RequestValidator.MAX_RECOMMENDATIONS:
            return False, f"n_recommendations cannot exceed {RequestValidator.MAX_RECOMMENDATIONS}, got {n_int}", None
        
        return True, None, n_int


def log_malformed_request(error_type: str, details: Dict) -> None:
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
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = kwargs.get('user_id')
        
        is_valid, error_msg, parsed_user_id = RequestValidator.validate_user_id(user_id)
        if not is_valid:
            log_malformed_request(
                'INVALID_USER_ID',
                {'user_id': user_id, 'error': error_msg, 'query_params': dict(request.args)}
            )
            return jsonify({
                'error': 'Invalid request',
                'message': error_msg,
                'user_id': user_id
            }), 400
        
        kwargs['user_id'] = str(parsed_user_id)
        
        n_recommendations = request.args.get('n', None)
        is_valid, error_msg, parsed_n = RequestValidator.validate_n_recommendations(n_recommendations)
        
        if not is_valid:
            log_malformed_request(
                'INVALID_PARAMETER',
                {'user_id': user_id, 'parameter': 'n_recommendations', 'value': n_recommendations, 'error': error_msg}
            )
            return jsonify({
                'error': 'Invalid request',
                'message': error_msg,
                'parameter': 'n_recommendations'
            }), 400
        
        kwargs['n_recommendations'] = parsed_n
        
        allowed_params = {'n'}
        unexpected_params = set(request.args.keys()) - allowed_params
        if unexpected_params:
            log_malformed_request(
                'UNEXPECTED_PARAMETERS',
                {'user_id': user_id, 'unexpected_params': list(unexpected_params), 'all_params': dict(request.args)}
            )
            logger.warning(f"Unexpected parameters in request: {unexpected_params}")
        
        return f(*args, **kwargs)
    
    return decorated_function


class RecommenderService:
    """Wrapper service for the trained SVD recommendation model."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        try:
            self.model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def get_recommendations(self, user_id: str, n_recommendations: int = 20) -> List[str]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self.model.predict(user_id, n_recommendations=n_recommendations)


def initialize_ab_testing():
    """Load Model A and Model B for A/B testing."""
    global recommender_A, recommender_B

    logger.info("Initializing A/B testing models...")

    try:
        if not os.path.exists(MODEL_A_PATH):
            logger.error(f"Model A file not found: {MODEL_A_PATH}")
            recommender_A = None
        else:
            recommender_A = RecommenderService(str(MODEL_A_PATH))
            logger.info(f" Model A loaded from {MODEL_A_PATH}")
    except Exception as e:
        logger.error(f"Failed loading Model A from {MODEL_A_PATH}: {e}")
        recommender_A = None

    try:
        if not os.path.exists(MODEL_B_PATH):
            logger.error(f"Model B file not found: {MODEL_B_PATH}")
            recommender_B = None
        else:
            recommender_B = RecommenderService(str(MODEL_B_PATH))
            logger.info(f" Model B loaded from {MODEL_B_PATH}")
    except Exception as e:
        logger.error(f"Failed loading Model B from {MODEL_B_PATH}: {e}")
        recommender_B = None

    if recommender_A is None and recommender_B is None:
        logger.critical("FATAL: No models loaded. API will fail all requests.")
        raise RuntimeError("Failed to load any models for A/B testing")
    elif recommender_A is None:
        logger.warning("Model A failed to load - all users will use Model B")
    elif recommender_B is None:
        logger.warning("Model B failed to load - all users will use Model A")
    else:
        logger.info(f" A/B testing initialized successfully")


def choose_model(user_id: str) -> str:
    """Deterministic 50/50 split using MD5 hash."""
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "A" if h % 2 == 0 else "B"


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'The request was malformed or invalid'
    }), 400


@app.errorhandler(404)
def not_found(error):
    log_malformed_request(
        'ENDPOINT_NOT_FOUND',
        {'path': request.path, 'method': request.method}
    )
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/health', '/recommend/<user_id>', '/experiment/stats', '/']
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
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
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'ab_testing_enabled': True,
        'model_a_loaded': recommender_A is not None and recommender_A.model is not None,
        'model_b_loaded': recommender_B is not None and recommender_B.model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/experiment/stats', methods=['GET'])
def experiment_stats():
    """
    NEW ENDPOINT: Get current A/B test statistics.
    
    Returns real-time stats on model performance.
    """
    stats = ab_logger.get_stats_summary()
    return jsonify({
        'experiment': 'A/B Test - Model Comparison',
        'statistics': stats,
        'timestamp': datetime.now().isoformat(),
        'note': 'Run analysis script for detailed statistical tests'
    })


@app.route('/experiment/reset', methods=['POST'])
def experiment_reset():
    """
    NEW ENDPOINT: Reset experiment statistics (not log file).
    
    Useful for starting a new experiment phase.
    """
    ab_logger.reset_stats()
    return jsonify({
        'message': 'Experiment statistics reset',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/recommend/<user_id>', methods=['GET'])
@validate_request
def recommend_movies(user_id: str, n_recommendations: int = 20):
    """Get movie recommendations with A/B test logging."""
    start_time = time.time()
    model_choice = None
    recommendations = []
    success = False
    error_msg = None

    try:
        if recommender_A is None and recommender_B is None:
            logger.warning("Models not initialized, initializing now")
            initialize_ab_testing()

        # Choose model
        model_choice = choose_model(user_id)
        model = recommender_A if model_choice == "A" else recommender_B

        # Fallback if chosen model failed
        if model is None:
            logger.warning(f"Model {model_choice} unavailable for user {user_id}, trying fallback")
            model = recommender_B if model_choice == "A" else recommender_A
            model_choice = "B" if model_choice == "A" else "A"
            
            if model is None:
                logger.error("Both models unavailable")
                error_msg = "Service unavailable - no models loaded"
                return jsonify({"error": error_msg}), 503

        # Get recommendations
        recommendations = model.get_recommendations(user_id, n_recommendations=n_recommendations)
        success = True

        result = ','.join(map(str, recommendations))
        response_time = time.time() - start_time

        # Log to standard logger
        logger.info(
            f"SUCCESS - user={user_id}, model={model_choice}, "
            f"n_requested={n_recommendations}, n_returned={len(recommendations)}, "
            f"response_time={response_time:.3f}s"
        )

        # Log to A/B test logger
        ab_logger.log_recommendation(
            user_id=user_id,
            model=model_choice,
            recommendations=recommendations,
            response_time=response_time,
            n_requested=n_recommendations,
            success=True
        )

        if response_time > 0.6:
            logger.warning(
                f"SLOW_RESPONSE: {response_time:.3f}s for user {user_id} (model={model_choice})"
            )

        return result, 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        response_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Unexpected error in recommend endpoint: {e}", exc_info=True)
        
        # Log failure to A/B test logger
        if model_choice:
            ab_logger.log_recommendation(
                user_id=user_id,
                model=model_choice,
                recommendations=[],
                response_time=response_time,
                n_requested=n_recommendations,
                success=False,
                error=error_msg
            )
        
        return jsonify({
            'error': 'Internal error',
            'message': 'Error generating recommendations'
        }), 500


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Movie Recommendation API with A/B Testing',
        'version': '2.1',
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
            },
            'experiment_stats': {
                'path': '/experiment/stats',
                'method': 'GET',
                'description': 'View A/B test statistics'
            },
            'experiment_reset': {
                'path': '/experiment/reset',
                'method': 'POST',
                'description': 'Reset experiment statistics'
            }
        },
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    try:
        initialize_ab_testing()
    except Exception as e:
        logger.critical(f"Failed to initialize A/B testing: {e}")
        exit(1)

    host = config.SERVING_CONFIG['host']
    port = config.SERVING_CONFIG['port']
    debug = config.SERVING_CONFIG['debug']

    logger.info(f"Starting Flask server on {host}:{port}")
    logger.info(f"Model A: {MODEL_A_PATH}")
    logger.info(f"Model B: {MODEL_B_PATH}")
    logger.info(f"Logging to: api_requests.log, malformed_requests.log, ab_test_results.jsonl")
    
    app.run(host=host, port=port, debug=debug)