"""
Online Accuracy Metric Computation

Calculates accuracy metrics from real production traffic.
Uses user interactions (clicks, watches) as implicit feedback.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats


class OnlineAccuracyCalculator:
    """
    Calculate online accuracy metrics from production traffic.
    
    Metrics:
    - Click-Through Rate (CTR): % of recommendations clicked
    - Watch-Through Rate (WTR): % of recommendations watched ≥20 min
    - Precision@K: % of top-K recommendations that were clicked/watched
    - NDCG@K: Normalized DCG considering position
    """
    
    def __init__(self,
                 recommendations_log: str = "ab_test_results.jsonl",
                 interactions_log: str = "user_interactions.jsonl"):
        """
        Initialize calculator.
        
        Args:
            recommendations_log: Log of recommendations served
            interactions_log: Log of user interactions
        """
        self.recommendations_log = Path(recommendations_log)
        self.interactions_log = Path(interactions_log)
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load recommendations and interactions."""
        recommendations = []
        interactions = []
        
        # Load recommendations
        if self.recommendations_log.exists():
            with open(self.recommendations_log, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('success') and entry.get('recommendations'):
                            recommendations.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        # Load interactions
        if self.interactions_log.exists():
            with open(self.interactions_log, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('event_type') == 'interaction':
                            interactions.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return recommendations, interactions
    
    def calculate_ctr(self, model: str = None) -> Dict:
        """
        Calculate Click-Through Rate.
        
        CTR = (Number of clicked recommendations) / (Total recommendations)
        
        Args:
            model: Calculate for specific model or None for all
            
        Returns:
            Dictionary with CTR metrics
        """
        recommendations, interactions = self.load_data()
        
        # Build interaction lookup: (user_id, movie_id) -> interaction
        clicks = set()
        for interaction in interactions:
            if interaction.get('interaction_type') == 'click':
                if model is None or interaction.get('model') == model:
                    clicks.add((interaction['user_id'], interaction['movie_id']))
        
        # Count total recommendations and clicks
        total_recs = 0
        clicked_recs = 0
        
        for rec in recommendations:
            if model is not None and rec.get('model') != model:
                continue
            
            user_id = rec['user_id']
            for movie_id in rec['recommendations']:
                total_recs += 1
                if (user_id, movie_id) in clicks:
                    clicked_recs += 1
        
        ctr = (clicked_recs / total_recs * 100) if total_recs > 0 else 0
        
        return {
            'metric': 'Click-Through Rate',
            'model': model if model else 'all',
            'total_recommendations': total_recs,
            'clicked_recommendations': clicked_recs,
            'ctr_percent': round(ctr, 2)
        }
    
    def calculate_precision_at_k(self, k: int = 10, model: str = None) -> Dict:
        """
        Calculate Precision@K from online interactions.
        
        Precision@K = (Relevant items in top-K) / K
        Where "relevant" = clicked or watched
        
        Args:
            k: Number of top recommendations to consider
            model: Calculate for specific model or None for all
            
        Returns:
            Dictionary with Precision@K metrics
        """
        recommendations, interactions = self.load_data()
        
        # Build interaction lookup
        interacted = set()
        for interaction in interactions:
            if model is None or interaction.get('model') == model:
                interacted.add((interaction['user_id'], interaction['movie_id']))
        
        # Calculate precision for each recommendation list
        precisions = []
        
        for rec in recommendations:
            if model is not None and rec.get('model') != model:
                continue
            
            user_id = rec['user_id']
            top_k = rec['recommendations'][:k]
            
            relevant_in_top_k = sum(
                1 for movie_id in top_k
                if (user_id, movie_id) in interacted
            )
            
            precision = relevant_in_top_k / k if k > 0 else 0
            precisions.append(precision)
        
        avg_precision = np.mean(precisions) if precisions else 0
        
        return {
            'metric': f'Precision@{k}',
            'model': model if model else 'all',
            'num_sessions': len(precisions),
            'avg_precision': round(avg_precision, 4),
            'std_precision': round(np.std(precisions), 4) if precisions else 0
        }
    
    def calculate_ndcg_at_k(self, k: int = 10, model: str = None) -> Dict:
        """
        Calculate NDCG@K from online interactions.
        
        Considers position of relevant items (earlier = better).
        
        Args:
            k: Number of top recommendations to consider
            model: Calculate for specific model or None for all
            
        Returns:
            Dictionary with NDCG@K metrics
        """
        recommendations, interactions = self.load_data()
        
        # Build interaction lookup with weights
        interaction_weights = {}
        for interaction in interactions:
            if model is None or interaction.get('model') == model:
                key = (interaction['user_id'], interaction['movie_id'])
                
                # Weight by interaction type
                if interaction['interaction_type'] == 'watch':
                    weight = 1.0  # Full relevance
                elif interaction['interaction_type'] == 'click':
                    weight = 0.5  # Partial relevance
                else:
                    weight = 0.3
                
                interaction_weights[key] = max(
                    interaction_weights.get(key, 0),
                    weight
                )
        
        # Calculate NDCG for each recommendation list
        ndcgs = []
        
        for rec in recommendations:
            if model is not None and rec.get('model') != model:
                continue
            
            user_id = rec['user_id']
            top_k = rec['recommendations'][:k]
            
            # Calculate DCG
            dcg = 0.0
            for i, movie_id in enumerate(top_k, 1):
                key = (user_id, movie_id)
                relevance = interaction_weights.get(key, 0)
                dcg += relevance / np.log2(i + 1)
            
            # Calculate ideal DCG (best possible ordering)
            relevances = [
                interaction_weights.get((user_id, movie_id), 0)
                for movie_id in top_k
            ]
            ideal_relevances = sorted(relevances, reverse=True)
            idcg = sum(
                rel / np.log2(i + 1)
                for i, rel in enumerate(ideal_relevances, 1)
            )
            
            # NDCG
            ndcg = (dcg / idcg) if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0
        
        return {
            'metric': f'NDCG@{k}',
            'model': model if model else 'all',
            'num_sessions': len(ndcgs),
            'avg_ndcg': round(avg_ndcg, 4),
            'std_ndcg': round(np.std(ndcgs), 4) if ndcgs else 0
        }
    
    def compare_models(self, metric: str = 'ctr', k: int = 10) -> Dict:
        """
        Compare Model A vs Model B using specified metric.
        
        Args:
            metric: 'ctr', 'precision', or 'ndcg'
            k: For precision/ndcg, the K value
            
        Returns:
            Comparison with statistical test
        """
        if metric == 'ctr':
            metrics_a = self.calculate_ctr('A')
            metrics_b = self.calculate_ctr('B')
            
            # Use proportions test for CTR
            n_a = metrics_a['total_recommendations']
            n_b = metrics_b['total_recommendations']
            x_a = metrics_a['clicked_recommendations']
            x_b = metrics_b['clicked_recommendations']
            
            # Two-proportion z-test
            if n_a > 0 and n_b > 0:
                p_a = x_a / n_a
                p_b = x_b / n_b
                p_pooled = (x_a + x_b) / (n_a + n_b)
                
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
                z_stat = (p_b - p_a) / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                p_value = 1.0
            
            return {
                'metric': 'CTR',
                'model_a': metrics_a,
                'model_b': metrics_b,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'difference': metrics_b['ctr_percent'] - metrics_a['ctr_percent']
            }
        
        elif metric == 'precision':
            # Get per-session precisions
            recommendations, interactions = self.load_data()
            
            # Calculate precision per session for each model
            precisions_a = []
            precisions_b = []
            
            # [Implementation similar to calculate_precision_at_k but returning list]
            
            # T-test on precisions
            if len(precisions_a) > 1 and len(precisions_b) > 1:
                t_stat, p_value = stats.ttest_ind(precisions_a, precisions_b)
            else:
                p_value = 1.0
            
            return {
                'metric': f'Precision@{k}',
                'model_a': {'mean': np.mean(precisions_a) if precisions_a else 0},
                'model_b': {'mean': np.mean(precisions_b) if precisions_b else 0},
                'p_value': p_value,
                'significant': p_value < 0.05
            }


if __name__ == "__main__":
    calc = OnlineAccuracyCalculator()
    
    # Calculate metrics for both models
    print("Model A CTR:", calc.calculate_ctr('A'))
    print("Model B CTR:", calc.calculate_ctr('B'))
    print()
    print("Model A Precision@10:", calc.calculate_precision_at_k(10, 'A'))
    print("Model B Precision@10:", calc.calculate_precision_at_k(10, 'B'))