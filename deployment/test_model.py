#!/usr/bin/env python3
"""
Test script to verify the SVD model works correctly
"""

import pickle
import sys
import os

# Add the src directory to path so we can import the model class
sys.path.append('./src')
from model_class import ImprovedSVDRecommendationModel

def test_model():
    """Test the SVD model loading and prediction"""

    model_path = './models/improved_svd_model_colab.pkl'

    print("Testing SVD Model...")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        print("ERROR: Model file not found!")
        return False

    try:
        # Load the model
        print("\nLoading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")

        # Check model attributes
        if hasattr(model, 'user_ids'):
            print(f"Number of users: {len(model.user_ids)}")
        if hasattr(model, 'all_movie_ids'):
            print(f"Number of movies: {len(model.all_movie_ids)}")
        if hasattr(model, 'popular_movies'):
            print(f"Popular movies available: {len(model.popular_movies)}")

        # Test prediction
        print("\nTesting predictions...")

        # Test with a user from the model
        if hasattr(model, 'user_ids') and len(model.user_ids) > 0:
            test_user = model.user_ids[0]
            print(f"Testing with user: {test_user}")

            recommendations = model.predict(test_user, n_recommendations=5)
            print(f"Recommendations: {recommendations}")

            if len(recommendations) > 0:
                print("✓ Model prediction successful!")
                return True
            else:
                print("✗ Model returned no recommendations")
                return False
        else:
            print("✗ No users found in model")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Model test FAILED!")
        sys.exit(1)