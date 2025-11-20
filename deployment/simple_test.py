#!/usr/bin/env python3
"""
Simple test of the Flask API service
"""

import sys
import os
import time
sys.path.append('./src')

from model_class import ImprovedSVDRecommendationModel
from app import SVDRecommenderWrapper, initialize_recommender, app

def test_api_components():
    """Test the API components locally"""

    print("Testing API components...")

    # Test 1: Model wrapper
    print("\n1. Testing SVDRecommenderWrapper...")
    wrapper = SVDRecommenderWrapper(
        './models/improved_svd_model_colab.pkl',
        './data/interactions.parquet'
    )

    if wrapper.model:
        print("   Model loaded successfully")

        # Test recommendation
        test_user = "100008"  # Use a known user from the model
        recommendations = wrapper.get_recommendations(test_user, 5)

        if recommendations and len(recommendations) > 0:
            print(f"   Recommendations for user {test_user}: {recommendations}")
            print("   Model wrapper test: PASSED")
        else:
            print("   No recommendations generated")
            return False
    else:
        print("   Model failed to load")
        return False

    # Test 2: Flask app endpoints
    print("\n2. Testing Flask app...")
    with app.test_client() as client:

        # Initialize the recommender
        initialize_recommender()

        # Test health endpoint
        response = client.get('/health')
        if response.status_code == 200:
            print("   Health endpoint: PASSED")
        else:
            print(f"   Health endpoint failed: {response.status_code}")
            return False

        # Test recommendation endpoint
        response = client.get('/recommend/100008')
        if response.status_code == 200:
            result = response.get_data(as_text=True)
            movies = result.split(',')
            if len(movies) > 0:
                print(f"   Recommendation endpoint: PASSED")
                print(f"   Response: {result[:100]}...")
                print(f"   Number of movies: {len(movies)}")
            else:
                print("   Empty recommendation response")
                return False
        else:
            print(f"   Recommendation endpoint failed: {response.status_code}")
            return False

    return True

if __name__ == "__main__":
    try:
        success = test_api_components()
        if success:
            print("\nAll tests PASSED! API is ready for deployment.")
        else:
            print("\nSome tests FAILED. Check the issues above.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()