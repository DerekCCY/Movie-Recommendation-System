"""
Data Drift Detection Module
===========================

Provides tools to monitor data distribution drift between
training (baseline) and recent production data.

Main Components
---------------
1. compute_baseline_statistics(df, output_path)
   → Computes & saves baseline statistics from training data

2. detect_drift(baseline_stats, recent_df)
   → Compares recent data with baseline using KS-test and PSI

3. calculate_psi_metric(baseline_values, recent_values)
   → Utility function to compute Population Stability Index (PSI)

4. load_baseline(path)
   → Helper to load a saved baseline JSON

Monitored distributions:
- Rating distribution  (KS test)
- User activity         (KS test)
- Movie popularity      (PSI)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------
# Data Type Handling
# ---------------------------------------------------------------------
def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python(x) for x in obj)
    else:
        return obj

# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# =================================================================
# DRIFT DETECTION (Maybe can be it's own module. For now here)
# =================================================================

def compute_baseline_statistics(interactions_df: pd.DataFrame, 
                                output_path: str = None) -> Dict[str, Any]:
    """
    Compute baseline statistics from training data for drift detection.
    
    Calculates distributions that will be monitored for drift:
    - Rating distribution
    - User activity distribution  
    - Movie popularity distribution
    
    Args:
        interactions_df: Training interactions DataFrame
        output_path: Optional path to save baseline stats as JSON
    
    Returns:
        Dict with baseline statistics
    
    Example:
        >>> baseline = compute_baseline_statistics(train_df, "data/baseline_stats.json")
        >>> print(baseline['rating_stats']['mean'])
        3.52
    """
    logger.info("Computing baseline statistics for drift detection...")
    
    baseline = {
        'created_at': pd.Timestamp.now().isoformat(),
        'n_interactions': int(len(interactions_df)),
        'n_users': int(interactions_df['user_id'].nunique()),
        'n_movies': int(interactions_df['movie_id'].nunique())
    }
    
    # Rating statistics
    if 'rating' in interactions_df.columns:
        ratings = interactions_df['rating'].dropna()
        baseline['rating_stats'] = {
            'mean': float(ratings.mean()),
            'std': float(ratings.std()),
            'median': float(ratings.median()),
            'distribution': {int(k): int(v) for k, v in ratings.value_counts().to_dict().items()},
            'percentiles': {
                '25': float(ratings.quantile(0.25)),
                '50': float(ratings.quantile(0.50)),
                '75': float(ratings.quantile(0.75)),
                '90': float(ratings.quantile(0.90))
            }
        }
        # Store raw ratings for KS test
        baseline['rating_values'] = ratings.tolist()
    
    # User activity statistics
    user_counts = interactions_df['user_id'].value_counts()
    baseline['user_activity_stats'] = {
        'mean_interactions_per_user': float(user_counts.mean()),
        'median_interactions_per_user': float(user_counts.median()),
        'std': float(user_counts.std()),
        'percentiles': {
            '25': float(user_counts.quantile(0.25)),
            '50': float(user_counts.quantile(0.50)),
            '75': float(user_counts.quantile(0.75)),
            '90': float(user_counts.quantile(0.90))
        }
    }
    # Store for statistical tests
    baseline['user_activity_values'] = [int(x) for x in user_counts.tolist()]
    
    # Movie popularity statistics
    movie_counts = interactions_df['movie_id'].value_counts()
    baseline['movie_popularity_stats'] = {
        'mean_interactions_per_movie': float(movie_counts.mean()),
        'median_interactions_per_movie': float(movie_counts.median()),
        'std': float(movie_counts.std()),
        'top_10_movies': [str(x) for x in movie_counts.head(10).index.tolist()],
        'top_10_counts': [int(x) for x in movie_counts.head(10).tolist()]
    }
    baseline['movie_popularity_values'] = [int(x) for x in movie_counts.tolist()]
    
    # Sparsity
    baseline['sparsity'] = float(1 - (len(interactions_df) / (baseline['n_users'] * baseline['n_movies'])))
    
    # Save to file if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        logger.info(f"Baseline statistics saved to {output_path}")
    
    logger.info(f"Baseline computed: {baseline['n_interactions']} interactions, "
               f"{baseline['n_users']} users, {baseline['n_movies']} movies")
    
    return baseline


def detect_drift(baseline_stats: Dict[str, Any], 
                recent_df: pd.DataFrame,
                significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Detect data drift by comparing recent data to baseline statistics.
    
    Uses statistical tests:
    - KS test for rating distribution
    - KS test for user activity distribution
    - PSI for movie popularity
    
    Args:
        baseline_stats: Baseline statistics from training data
        recent_df: Recent production data DataFrame
        significance_level: P-value threshold (default 0.05)
    
    Returns:
        Dict with drift detection results for each monitored distribution
    
    Example:
        >>> baseline = load_baseline("data/baseline_stats.json")
        >>> drift_report = detect_drift(baseline, recent_data)
        >>> if drift_report['rating_distribution']['drift_detected']:
        ...     print("WARNING: Rating drift detected!")
    """
    from scipy import stats as scipy_stats
    import numpy as np
    
    logger.info("Detecting data drift...")
    
    drift_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'recent_data_size': len(recent_df),
        'baseline_data_size': baseline_stats['n_interactions'],
        'tests_performed': [],
        'drift_detected': False,
        'drift_summary': []
    }
    
    # =================================================================
    # 1. RATING DISTRIBUTION DRIFT (KS Test)
    # =================================================================
    if 'rating' in recent_df.columns and 'rating_values' in baseline_stats:
        logger.info("Testing rating distribution drift (KS test)...")
        
        baseline_ratings = np.array(baseline_stats['rating_values'])
        recent_ratings = recent_df['rating'].dropna().values
        
        if len(recent_ratings) > 0:
            # KS test
            ks_stat, p_value = scipy_stats.ks_2samp(baseline_ratings, recent_ratings)
            drift_detected = p_value < significance_level
            
            # Calculate distributional changes
            baseline_mean = baseline_stats['rating_stats']['mean']
            recent_mean = float(recent_ratings.mean())
            mean_shift = recent_mean - baseline_mean
            
            rating_drift = {
                'test': 'Kolmogorov-Smirnov',
                'metric': 'rating_distribution',
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'significance_level': significance_level,
                'drift_detected': drift_detected,
                'baseline_mean': baseline_mean,
                'recent_mean': recent_mean,
                'mean_shift': mean_shift,
                'interpretation': (
                    f"Rating distribution: KS={ks_stat:.4f}, p={p_value:.4f}. "
                    f"Mean shifted from {baseline_mean:.2f} to {recent_mean:.2f} ({mean_shift:+.2f}). "
                    f"{'DRIFT DETECTED' if drift_detected else 'No significant drift'}"
                )
            }
            
            drift_report['rating_distribution'] = rating_drift
            drift_report['tests_performed'].append('rating_distribution_ks')
            
            if drift_detected:
                drift_report['drift_detected'] = True
                drift_report['drift_summary'].append('Rating distribution has drifted')
                logger.warning(f"DRIFT DETECTED in rating distribution (p={p_value:.4f})")
    
    # =================================================================
    # 2. USER ACTIVITY DRIFT (KS Test)
    # =================================================================
    if 'user_activity_values' in baseline_stats:
        logger.info("Testing user activity drift (KS test)...")
        
        baseline_activity = np.array(baseline_stats['user_activity_values'])
        recent_activity = recent_df['user_id'].value_counts().values
        
        if len(recent_activity) > 0:
            # KS test
            ks_stat, p_value = scipy_stats.ks_2samp(baseline_activity, recent_activity)
            drift_detected = p_value < significance_level
            
            baseline_mean = baseline_stats['user_activity_stats']['mean_interactions_per_user']
            recent_mean = float(recent_activity.mean())
            mean_shift = recent_mean - baseline_mean
            
            activity_drift = {
                'test': 'Kolmogorov-Smirnov',
                'metric': 'user_activity',
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'significance_level': significance_level,
                'drift_detected': drift_detected,
                'baseline_mean': baseline_mean,
                'recent_mean': recent_mean,
                'mean_shift': mean_shift,
                'interpretation': (
                    f"User activity: KS={ks_stat:.4f}, p={p_value:.4f}. "
                    f"Mean interactions shifted from {baseline_mean:.1f} to {recent_mean:.1f} ({mean_shift:+.1f}). "
                    f"{'DRIFT DETECTED' if drift_detected else 'No significant drift'}"
                )
            }
            
            drift_report['user_activity'] = activity_drift
            drift_report['tests_performed'].append('user_activity_ks')
            
            if drift_detected:
                drift_report['drift_detected'] = True
                drift_report['drift_summary'].append('User activity pattern has drifted')
                logger.warning(f"DRIFT DETECTED in user activity (p={p_value:.4f})")
    
    # =================================================================
    # 3. MOVIE POPULARITY DRIFT (PSI)
    # =================================================================
    if 'movie_popularity_values' in baseline_stats:
        logger.info("Testing movie popularity drift (PSI)...")
        
        baseline_popularity = np.array(baseline_stats['movie_popularity_values'])
        recent_popularity = recent_df['movie_id'].value_counts().values
        
        if len(recent_popularity) > 0:
            # Calculate PSI
            psi = calculate_psi_metric(baseline_popularity, recent_popularity)
            
            # PSI thresholds: <0.1 no drift, 0.1-0.2 moderate, >0.2 significant
            if psi < 0.1:
                drift_level = 'no_drift'
                drift_detected = False
            elif psi < 0.2:
                drift_level = 'moderate_drift'
                drift_detected = True
            else:
                drift_level = 'significant_drift'
                drift_detected = True
            
            popularity_drift = {
                'test': 'Population Stability Index',
                'metric': 'movie_popularity',
                'psi': float(psi),
                'drift_level': drift_level,
                'drift_detected': drift_detected,
                'interpretation': (
                    f"Movie popularity: PSI={psi:.4f}. "
                    f"{'No significant drift' if psi < 0.1 else 'Moderate drift' if psi < 0.2 else 'Significant drift detected'}"
                )
            }
            
            drift_report['movie_popularity'] = popularity_drift
            drift_report['tests_performed'].append('movie_popularity_psi')
            
            if drift_detected:
                drift_report['drift_detected'] = True
                drift_report['drift_summary'].append(f'Movie popularity has {drift_level}')
                logger.warning(f"DRIFT DETECTED in movie popularity (PSI={psi:.4f})")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION REPORT")
    logger.info("=" * 60)
    logger.info(f"Tests performed: {len(drift_report['tests_performed'])}")
    logger.info(f"Overall drift detected: {drift_report['drift_detected']}")
    
    if drift_report['drift_summary']:
        logger.info("\nDrift Issues:")
        for issue in drift_report['drift_summary']:
            logger.info(f"  - {issue}")
    else:
        logger.info("\nNo significant drift detected")
    
    logger.info("=" * 60)
    
    return drift_report


def calculate_psi_metric(baseline_values, recent_values, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI thresholds:
    - PSI < 0.1: No significant drift
    - 0.1 <= PSI < 0.2: Moderate drift
    - PSI >= 0.2: Significant drift
    
    Args:
        baseline_values: Baseline distribution values
        recent_values: Recent distribution values
        bins: Number of bins for discretization
    
    Returns:
        PSI value
    """
    import numpy as np
    
    # Create bins based on baseline
    bin_edges = np.histogram_bin_edges(baseline_values, bins=bins)
    
    # Calculate proportions
    baseline_counts, _ = np.histogram(baseline_values, bins=bin_edges)
    recent_counts, _ = np.histogram(recent_values, bins=bin_edges)
    
    # Convert to proportions
    baseline_props = baseline_counts / len(baseline_values)
    recent_props = recent_counts / len(recent_values)
    
    # Avoid log(0)
    baseline_props = np.maximum(baseline_props, 1e-6)
    recent_props = np.maximum(recent_props, 1e-6)
    
    # Calculate PSI
    psi = np.sum((recent_props - baseline_props) * np.log(recent_props / baseline_props))
    
    return psi
