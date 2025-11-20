"""
A/B Test Experimentation Logger

Logs all recommendations with model assignments for later analysis.
Enables statistical comparison of model performance.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading
from collections import defaultdict


class ABTestLogger:
    """
    Thread-safe logger for A/B test experiments.
    Tracks which model served which user and enables statistical analysis.
    """
    
    def __init__(self, log_path: str = "ab_test_results.jsonl"):
        """
        Initialize the A/B test logger.
        
        Args:
            log_path: Path to JSONL log file
        """
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        
        # In-memory stats for quick access
        self.stats = {
            'A': defaultdict(int),
            'B': defaultdict(int)
        }
        
    def log_recommendation(self, 
                          user_id: str,
                          model: str,
                          recommendations: List[str],
                          response_time: float,
                          n_requested: int,
                          success: bool = True,
                          error: Optional[str] = None) -> None:
        """
        Log a recommendation request with all relevant metadata.
        
        Args:
            user_id: User who received recommendations
            model: Which model was used ("A" or "B")
            recommendations: List of recommended movie IDs
            response_time: Time taken to generate recommendations (seconds)
            n_requested: Number of recommendations requested
            success: Whether the request succeeded
            error: Error message if failed
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'model': model,
            'n_requested': n_requested,
            'n_returned': len(recommendations),
            'response_time_ms': round(response_time * 1000, 2),
            'success': success,
            'recommendations': recommendations[:10],  # Store first 10 for space
            'error': error
        }
        
        # Write to file (thread-safe)
        with self.lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            # Update in-memory stats
            self.stats[model]['total_requests'] += 1
            if success:
                self.stats[model]['successful_requests'] += 1
                self.stats[model]['total_response_time'] += response_time
                self.stats[model]['total_items_returned'] += len(recommendations)
            else:
                self.stats[model]['failed_requests'] += 1
    
    def log_user_feedback(self,
                         user_id: str,
                         movie_id: str,
                         rating: float,
                         model: Optional[str] = None) -> None:
        """
        Log user feedback (e.g., ratings on recommended movies).
        
        Args:
            user_id: User who gave feedback
            movie_id: Movie that was rated
            rating: Rating value
            model: Which model recommended this (if known)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'feedback',
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'model': model
        }
        
        with self.lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def get_stats_summary(self) -> Dict:
        """
        Get current statistics summary for both models.
        
        Returns:
            Dictionary with stats for models A and B
        """
        summary = {}
        
        for model in ['A', 'B']:
            stats = self.stats[model]
            total = stats['total_requests']
            successful = stats['successful_requests']
            
            if total > 0:
                summary[f'model_{model}'] = {
                    'total_requests': total,
                    'successful_requests': successful,
                    'failed_requests': stats['failed_requests'],
                    'success_rate': round(successful / total * 100, 2),
                    'avg_response_time_ms': round(
                        stats['total_response_time'] / successful * 1000, 2
                    ) if successful > 0 else 0,
                    'avg_items_per_request': round(
                        stats['total_items_returned'] / successful, 2
                    ) if successful > 0 else 0
                }
            else:
                summary[f'model_{model}'] = {
                    'total_requests': 0,
                    'message': 'No requests logged yet'
                }
        
        return summary
    
    def reset_stats(self) -> None:
        """Reset in-memory statistics (doesn't affect log file)."""
        with self.lock:
            self.stats = {
                'A': defaultdict(int),
                'B': defaultdict(int)
            }


# Global logger instance
ab_logger = ABTestLogger()