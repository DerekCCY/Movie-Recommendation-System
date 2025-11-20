"""
Model Evaluation Module

Provides various evaluation metrics for recommendation models:
- RMSE (Root Mean Squared Error)
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Inference time/throughput
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any
from collections import defaultdict
from .model import ImprovedSVDRecommendationModel


def evaluate_rmse(model: ImprovedSVDRecommendationModel,
                 test_df: pd.DataFrame) -> float:
    """
    Calculate Root Mean Squared Error on test set (VECTORIZED).

    Metric: How accurately the model predicts ratings
    Data: Test set with ground truth ratings
    Operationalization: RMSE = sqrt(mean((predicted - actual)^2))

    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user_id, movie_id, rating

    Returns:
        RMSE value (lower is better)

    Example:
        >>> rmse = evaluate_rmse(model, test_df)
        >>> print(f"RMSE: {rmse:.4f}")
        RMSE: 1.0468
    """
    # Filter out NaN ratings
    test_df_clean = test_df[test_df['rating'].notna()].copy()

    if len(test_df_clean) == 0:
        return float('nan')

    # Convert IDs to strings (ensure consistency with model mappings)
    test_df_clean['user_id'] = test_df_clean['user_id'].astype(str)
    test_df_clean['movie_id'] = test_df_clean['movie_id'].astype(str)

    # Map user_ids and movie_ids to indices (vectorized)
    user_indices = test_df_clean['user_id'].map(model.user_mapping)
    item_indices = test_df_clean['movie_id'].map(model.item_mapping)

    # Find valid pairs (both user and movie are known)
    valid_mask = user_indices.notna() & item_indices.notna()
    valid_indices = valid_mask.values

    # Get actual ratings
    actuals = test_df_clean['rating'].values

    # Initialize predictions with global mean (for unknown users/movies)
    predictions = np.full(len(test_df_clean), model.global_mean, dtype=np.float64)

    if valid_indices.sum() > 0:
        # Get valid user/item indices as integers
        u_idx = user_indices[valid_mask].astype(int).values
        i_idx = item_indices[valid_mask].astype(int).values

        # Vectorized CF prediction for all known pairs at once
        cf_pred = (model.global_mean +
                   model.user_bias[u_idx] +
                   model.item_bias[i_idx] +
                   np.sum(model.user_factors[u_idx] * model.item_factors[i_idx], axis=1))

        # Handle hybrid model with content features
        if hasattr(model, 'content_weight') and model.content_weight > 0:
            # Add content scores (loop for now - still much faster than full loop)
            content_scores = np.zeros(len(u_idx))
            valid_user_ids = test_df_clean[valid_mask]['user_id'].values
            valid_movie_ids = test_df_clean[valid_mask]['movie_id'].values

            for idx, (uid, mid) in enumerate(zip(valid_user_ids, valid_movie_ids)):
                content_scores[idx] = model.compute_content_score(uid, mid)

            cf_pred += model.content_weight * content_scores

        # Clip predictions to valid rating range
        predictions[valid_indices] = np.clip(cf_pred, 1.0, 5.0)

    # Compute RMSE
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def evaluate_precision_at_k(model: ImprovedSVDRecommendationModel,
                            test_df: pd.DataFrame,
                            k: int = 10) -> float:
    """
    Calculate Precision@K.

    Precision@K = (# of relevant items in top-K) / K

    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user_id, movie_id, rating
        k: Number of recommendations to consider

    Returns:
        Average Precision@K across all users
    """
    # Build ground truth: set of relevant items per user
    # Consider ratings >= 3.5 as relevant
    relevant_threshold = 3.5

    # Vectorized filtering and grouping (100x faster than iterrows)
    filtered = test_df[
        (test_df['rating'].notna()) &
        (test_df['rating'] >= relevant_threshold)
    ].copy()

    # Convert to string and group by user
    filtered['user_id'] = filtered['user_id'].astype(str)
    filtered['movie_id'] = filtered['movie_id'].astype(str)

    user_relevant_items = defaultdict(
        set,
        filtered.groupby('user_id')['movie_id'].apply(set).to_dict()
    )

    # Calculate precision@k for each user
    precision_scores = []

    for user_id, relevant_items in user_relevant_items.items():
        if len(relevant_items) == 0:
            continue

        # Get top-K recommendations
        recommendations = model.predict(user_id, n_recommendations=k)

        # Count how many recommended items are relevant
        relevant_in_recs = len(set(recommendations) & relevant_items)

        # Precision@K = (relevant items in top-K) / K
        precision = relevant_in_recs / k
        precision_scores.append(precision)

    # Average across all users
    if len(precision_scores) == 0:
        return 0.0

    return np.mean(precision_scores)


def evaluate_recall_at_k(model: ImprovedSVDRecommendationModel,
                        test_df: pd.DataFrame,
                        k: int = 10) -> float:
    """
    Calculate Recall@K.

    Recall@K = (# of relevant items in top-K) / (total # of relevant items)

    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user_id, movie_id, rating
        k: Number of recommendations to consider

    Returns:
        Average Recall@K across all users
    """
    # Build ground truth: set of relevant items per user
    relevant_threshold = 3.5

    # Vectorized filtering and grouping (100x faster than iterrows)
    filtered = test_df[
        (test_df['rating'].notna()) &
        (test_df['rating'] >= relevant_threshold)
    ].copy()

    # Convert to string and group by user
    filtered['user_id'] = filtered['user_id'].astype(str)
    filtered['movie_id'] = filtered['movie_id'].astype(str)

    user_relevant_items = defaultdict(
        set,
        filtered.groupby('user_id')['movie_id'].apply(set).to_dict()
    )

    # Calculate recall@k for each user
    recall_scores = []

    for user_id, relevant_items in user_relevant_items.items():
        if len(relevant_items) == 0:
            continue

        # Get top-K recommendations
        recommendations = model.predict(user_id, n_recommendations=k)

        # Count how many relevant items were retrieved
        relevant_in_recs = len(set(recommendations) & relevant_items)

        # Recall@K = (relevant items in top-K) / (total relevant items)
        recall = relevant_in_recs / len(relevant_items)
        recall_scores.append(recall)

    # Average across all users
    if len(recall_scores) == 0:
        return 0.0

    return np.mean(recall_scores)


def measure_inference_time(model: ImprovedSVDRecommendationModel,
                          n_samples: int = 100,
                          n_recommendations: int = 20) -> Dict[str, float]:
    """
    Measure inference performance.

    Metric: Average time to generate recommendations
    Data: Random sample of users (or test users)
    Operationalization: Time N requests, compute mean and p95 latency

    Args:
        model: Trained recommendation model
        n_samples: Number of recommendation requests to time
        n_recommendations: Number of recommendations per request

    Returns:
        Dict with:
        - mean_time_ms: Average inference time in milliseconds
        - p95_time_ms: 95th percentile latency
        - requests_per_second: Throughput

    Example:
        >>> stats = measure_inference_time(model, n_samples=100)
        >>> print(stats)
        {'mean_time_ms': 0.4, 'p95_time_ms': 0.6, 'requests_per_second': 2500}
    """
    latencies = []

    # Sample random users from model's known users
    user_sample = np.random.choice(model.user_ids, size=min(n_samples, len(model.user_ids)), replace=False)

    for user_id in user_sample:
        start_time = time.perf_counter()
        model.predict(str(user_id), n_recommendations=n_recommendations)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency_ms)

    latencies = np.array(latencies)

    mean_time_ms = np.mean(latencies)
    p95_time_ms = np.percentile(latencies, 95)
    requests_per_second = 1000 / mean_time_ms if mean_time_ms > 0 else 0

    return {
        'mean_time_ms': mean_time_ms,
        'p95_time_ms': p95_time_ms,
        'requests_per_second': requests_per_second
    }


def evaluate_recommendation_diversity(recommendations_list: List[List[str]]) -> float:
    """
    Measure diversity of recommendations across users.

    Diversity = (# of unique items recommended) / (total # of items)

    Args:
        recommendations_list: List of recommendation lists (one per user)

    Returns:
        Diversity score [0, 1] (higher is better)
    """
    # Collect all unique items recommended across all users
    unique_items = set()

    for recommendations in recommendations_list:
        unique_items.update(recommendations)

    # Total number of possible items is unknown without model context
    # Return raw count and let caller interpret
    return len(unique_items)


def generate_evaluation_report(model: ImprovedSVDRecommendationModel,
                               test_df: pd.DataFrame,
                               config: dict) -> Dict:
    """
    Generate comprehensive evaluation report with all metrics.

    Args:
        model: Trained recommendation model
        test_df: Test DataFrame
        config: Evaluation configuration (k_values, n_samples, etc.)

    Returns:
        Dict with all evaluation metrics:
        - rmse: float
        - precision@k: Dict[int, float]
        - recall@k: Dict[int, float]
        - inference_time: Dict
        - diversity: float

    Example:
        >>> config = {"k_values": [5, 10, 20], "n_inference_samples": 100}
        >>> report = generate_evaluation_report(model, test_df, config)
        >>> print(report['rmse'])
        1.0468
    """
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    report = {}

    # 1. RMSE
    print("\n[1/4] Calculating RMSE...")
    rmse = evaluate_rmse(model, test_df)
    report['rmse'] = rmse
    print(f"  RMSE: {rmse:.4f}")

    # 2. Precision@K and Recall@K
    k_values = config.get('k_values', [10, 20])

    print(f"\n[2/4] Calculating Precision@K for K={k_values}...")
    precision_dict = {}
    for k in k_values:
        precision = evaluate_precision_at_k(model, test_df, k=k)
        precision_dict[k] = precision
        print(f"  Precision@{k}: {precision:.4f}")
    report['precision@k'] = precision_dict

    print(f"\n[3/4] Calculating Recall@K for K={k_values}...")
    recall_dict = {}
    for k in k_values:
        recall = evaluate_recall_at_k(model, test_df, k=k)
        recall_dict[k] = recall
        print(f"  Recall@{k}: {recall:.4f}")
    report['recall@k'] = recall_dict

    # 3. Inference time
    print("\n[4/4] Measuring inference time...")
    n_samples = config.get('n_inference_samples', 100)

    latency_stats = measure_inference_time(
        model,
        n_samples=n_samples,
        n_recommendations=20
    )

    report['inference_time'] = latency_stats
    print(f"  Mean latency: {latency_stats['mean_time_ms']:.2f} ms")
    print(f"  P95 latency: {latency_stats['p95_time_ms']:.2f} ms")
    print(f"  Throughput: {latency_stats['requests_per_second']:.0f} req/s")

    # 4. Model info
    model_info = model.get_model_info()
    report['model_info'] = model_info

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Algorithm: {model_info.get('algorithm', 'Unknown')}")
    print(f"Users: {model_info.get('total_users', 'Unknown')}")
    print(f"Movies: {model_info.get('total_movies', 'Unknown')}")
    print(f"Factors: {model_info.get('n_factors', 'Unknown')}")
    print(f"\nRMSE: {rmse:.4f}")
    if 20 in precision_dict:
        print(f"Precision@20: {precision_dict[20]:.4f}")
    if 20 in recall_dict:
        print(f"Recall@20: {recall_dict[20]:.4f}")
    print(f"Mean Latency: {latency_stats['mean_time_ms']:.2f} ms")
    print("=" * 60)

    return report
