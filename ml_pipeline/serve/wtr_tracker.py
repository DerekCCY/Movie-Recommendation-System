"""
Watch-Through Rate (WTR) Tracking and Analysis

Tracks recommendations and calculates WTR when watch data becomes available.
WTR = (Movies watched ≥20 min within 7 days) / (Total recommendations) × 100%
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading


class WTRTracker:
    """
    Track recommendations and watch events to calculate Watch-Through Rate.
    """
    
    def __init__(self, 
                 recommendations_log: str = "recommendations_log.jsonl",
                 watch_events_log: str = "watch_events_log.jsonl"):
        """
        Initialize WTR tracker.
        
        Args:
            recommendations_log: Path to log file for recommendations
            watch_events_log: Path to log file for watch events
        """
        self.recommendations_log = Path(recommendations_log)
        self.watch_events_log = Path(watch_events_log)
        self.lock = threading.Lock()
        
    def log_recommendation(self,
                          user_id: str,
                          movie_id: str,
                          model: str,
                          position: int,
                          session_id: Optional[str] = None) -> None:
        """
        Log a movie recommendation.
        
        Args:
            user_id: User who received recommendation
            movie_id: Movie that was recommended
            model: Which model made the recommendation ("A" or "B")
            position: Position in recommendation list (1-20)
            session_id: Optional session identifier
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'recommendation',
            'user_id': user_id,
            'movie_id': movie_id,
            'model': model,
            'position': position,
            'session_id': session_id
        }
        
        with self.lock:
            with open(self.recommendations_log, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def log_watch_event(self,
                       user_id: str,
                       movie_id: str,
                       watch_duration_minutes: float,
                       watched_at: Optional[datetime] = None) -> None:
        """
        Log a watch event (when user actually watches a movie).
        
        Args:
            user_id: User who watched
            movie_id: Movie that was watched
            watch_duration_minutes: How long they watched (in minutes)
            watched_at: When they watched (defaults to now)
        """
        if watched_at is None:
            watched_at = datetime.now()
        
        entry = {
            'timestamp': watched_at.isoformat(),
            'event_type': 'watch',
            'user_id': user_id,
            'movie_id': movie_id,
            'watch_duration_minutes': watch_duration_minutes
        }
        
        with self.lock:
            with open(self.watch_events_log, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def calculate_wtr(self,
                     model: Optional[str] = None,
                     time_window_days: int = 7,
                     min_watch_minutes: int = 20) -> Dict:
        """
        Calculate Watch-Through Rate for recommendations.
        
        Args:
            model: Calculate for specific model ("A" or "B") or None for all
            time_window_days: Days after recommendation to count watches
            min_watch_minutes: Minimum watch duration to count as "watched"
            
        Returns:
            Dictionary with WTR metrics
        """
        # Load recommendations
        recommendations = []
        if self.recommendations_log.exists():
            with open(self.recommendations_log, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get('event_type') == 'recommendation':
                            if model is None or rec.get('model') == model:
                                rec['timestamp'] = datetime.fromisoformat(rec['timestamp'])
                                recommendations.append(rec)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        # Load watch events
        watches = []
        if self.watch_events_log.exists():
            with open(self.watch_events_log, 'r') as f:
                for line in f:
                    try:
                        watch = json.loads(line)
                        if watch.get('event_type') == 'watch':
                            watch['timestamp'] = datetime.fromisoformat(watch['timestamp'])
                            watches.append(watch)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        # Build watch lookup: (user_id, movie_id) -> watch event
        watch_lookup = {}
        for watch in watches:
            key = (watch['user_id'], watch['movie_id'])
            # Keep the watch with longest duration
            if key not in watch_lookup or watch['watch_duration_minutes'] > watch_lookup[key]['watch_duration_minutes']:
                watch_lookup[key] = watch
        
        # Calculate WTR
        total_recommendations = len(recommendations)
        watched_count = 0
        watched_within_window = 0
        total_watch_duration = 0
        
        for rec in recommendations:
            key = (rec['user_id'], rec['movie_id'])
            
            if key in watch_lookup:
                watch = watch_lookup[key]
                watch_duration = watch['watch_duration_minutes']
                
                # Check if watched enough
                if watch_duration >= min_watch_minutes:
                    watched_count += 1
                    total_watch_duration += watch_duration
                    
                    # Check if within time window
                    time_diff = watch['timestamp'] - rec['timestamp']
                    if time_diff <= timedelta(days=time_window_days):
                        watched_within_window += 1
        
        # Calculate metrics
        wtr = (watched_within_window / total_recommendations * 100) if total_recommendations > 0 else 0
        watch_rate = (watched_count / total_recommendations * 100) if total_recommendations > 0 else 0
        avg_watch_duration = (total_watch_duration / watched_count) if watched_count > 0 else 0
        
        return {
            'model': model if model else 'all',
            'total_recommendations': total_recommendations,
            'watched_count': watched_count,
            'watched_within_window': watched_within_window,
            'wtr_percent': round(wtr, 2),
            'watch_rate_percent': round(watch_rate, 2),
            'avg_watch_duration_minutes': round(avg_watch_duration, 2),
            'time_window_days': time_window_days,
            'min_watch_minutes': min_watch_minutes
        }
    
    def compare_models_wtr(self,
                          time_window_days: int = 7,
                          min_watch_minutes: int = 20) -> Dict:
        """
        Compare WTR between Model A and Model B.
        
        Returns:
            Dictionary with comparison metrics
        """
        metrics_a = self.calculate_wtr('A', time_window_days, min_watch_minutes)
        metrics_b = self.calculate_wtr('B', time_window_days, min_watch_minutes)
        
        wtr_diff = metrics_b['wtr_percent'] - metrics_a['wtr_percent']
        wtr_lift = (wtr_diff / metrics_a['wtr_percent'] * 100) if metrics_a['wtr_percent'] > 0 else 0
        
        return {
            'model_a': metrics_a,
            'model_b': metrics_b,
            'comparison': {
                'wtr_difference': round(wtr_diff, 2),
                'wtr_lift_percent': round(wtr_lift, 2),
                'better_model': 'B' if wtr_diff > 0 else 'A' if wtr_diff < 0 else 'tie'
            }
        }
    
    def generate_wtr_report(self) -> str:
        """
        Generate a comprehensive WTR report.
        
        Returns:
            Formatted report string
        """
        comparison = self.compare_models_wtr()
        
        report = []
        report.append("=" * 70)
        report.append("WATCH-THROUGH RATE (WTR) ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        # Model A
        report.append("MODEL A")
        report.append("-" * 70)
        a = comparison['model_a']
        report.append(f"Total Recommendations:        {a['total_recommendations']}")
        report.append(f"Watched (≥{a['min_watch_minutes']} min):            {a['watched_count']} ({a['watch_rate_percent']}%)")
        report.append(f"Watched within {a['time_window_days']} days:        {a['watched_within_window']}")
        report.append(f"Watch-Through Rate (WTR):     {a['wtr_percent']}%")
        report.append(f"Avg Watch Duration:           {a['avg_watch_duration_minutes']} minutes")
        report.append("")
        
        # Model B
        report.append("MODEL B")
        report.append("-" * 70)
        b = comparison['model_b']
        report.append(f"Total Recommendations:        {b['total_recommendations']}")
        report.append(f"Watched (≥{b['min_watch_minutes']} min):            {b['watched_count']} ({b['watch_rate_percent']}%)")
        report.append(f"Watched within {b['time_window_days']} days:        {b['watched_within_window']}")
        report.append(f"Watch-Through Rate (WTR):     {b['wtr_percent']}%")
        report.append(f"Avg Watch Duration:           {b['avg_watch_duration_minutes']} minutes")
        report.append("")
        
        # Comparison
        report.append("COMPARISON")
        report.append("-" * 70)
        comp = comparison['comparison']
        report.append(f"WTR Difference:               {comp['wtr_difference']:+.2f} percentage points")
        report.append(f"WTR Lift:                     {comp['wtr_lift_percent']:+.2f}%")
        report.append(f"Better Model:                 Model {comp['better_model'].upper()}")
        report.append("")
        
        # Recommendation
        if abs(comp['wtr_difference']) > 2:  # If difference > 2 percentage points
            report.append(f"RECOMMENDATION: Model {comp['better_model'].upper()} shows meaningfully higher WTR")
            report.append(f"                ({abs(comp['wtr_difference']):.1f} percentage points improvement)")
        else:
            report.append("RECOMMENDATION: Models show similar WTR performance")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


# Global tracker instance
wtr_tracker = WTRTracker()


def simulate_watch_data(recommendations_log: str = "ab_test_results.jsonl",
                       watch_probability: float = 0.15,
                       high_position_boost: float = 2.0) -> None:
    """
    Simulate watch events for demonstration purposes.
    
    In production, this would come from your analytics pipeline.
    
    Args:
        recommendations_log: Log file with recommendations
        watch_probability: Base probability a recommendation is watched
        high_position_boost: Multiplier for top-3 recommendations
    """
    import random
    
    print("Simulating watch data for demonstration...")
    print(f"Base watch probability: {watch_probability * 100}%")
    print()
    
    # Load recommendations from A/B test log
    recommendations = []
    if Path(recommendations_log).exists():
        with open(recommendations_log, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('success') and entry.get('recommendations'):
                        for i, movie_id in enumerate(entry['recommendations'][:20], 1):
                            recommendations.append({
                                'user_id': entry['user_id'],
                                'movie_id': movie_id,
                                'model': entry['model'],
                                'position': i,
                                'timestamp': datetime.fromisoformat(entry['timestamp'])
                            })
                except (json.JSONDecodeError, KeyError):
                    continue
    
    print(f"Found {len(recommendations)} recommendations")
    
    # Log all recommendations to WTR tracker
    for rec in recommendations:
        wtr_tracker.log_recommendation(
            user_id=rec['user_id'],
            movie_id=rec['movie_id'],
            model=rec['model'],
            position=rec['position']
        )
    
    # Simulate watches
    watch_count = 0
    for rec in recommendations:
        # Higher positions more likely to be watched
        position_multiplier = high_position_boost if rec['position'] <= 3 else 1.0
        watch_prob = watch_probability * position_multiplier
        
        if random.random() < watch_prob:
            # Simulate watch duration (20-120 minutes)
            watch_duration = random.uniform(20, 120)
            
            # Simulate watch happening 0-7 days after recommendation
            days_later = random.randint(0, 7)
            watched_at = rec['timestamp'] + timedelta(days=days_later)
            
            wtr_tracker.log_watch_event(
                user_id=rec['user_id'],
                movie_id=rec['movie_id'],
                watch_duration_minutes=watch_duration,
                watched_at=watched_at
            )
            watch_count += 1
    
    print(f"Simulated {watch_count} watch events ({watch_count/len(recommendations)*100:.1f}% WTR)")
    print()


if __name__ == "__main__":
    # Example usage
    print("Watch-Through Rate Tracker")
    print()
    
    # Check if we have A/B test data
    if not Path("ab_test_results.jsonl").exists():
        print("No A/B test data found. Run experiment first:")
        print("  python -m ml_pipeline.serve.test_simulation --requests 500")
        exit(1)
    
    # Simulate watch data
    simulate_watch_data()
    
    # Generate report
    print(wtr_tracker.generate_wtr_report())