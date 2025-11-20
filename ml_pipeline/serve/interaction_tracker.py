"""
User Interaction Tracker

Logs user responses (clicks, watches, ratings) to recommendations.
This satisfies the "user response (click/interaction)" requirement.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading


class InteractionTracker:
    """
    Track user interactions with recommendations.
    Enables calculation of online accuracy metrics.
    """
    
    def __init__(self, log_path: str = "user_interactions.jsonl"):
        """
        Initialize interaction tracker.
        
        Args:
            log_path: Path to interaction log file
        """
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
    
    def log_interaction(self,
                       user_id: str,
                       movie_id: str,
                       interaction_type: str,
                       model: Optional[str] = None,
                       session_id: Optional[str] = None,
                       metadata: Optional[dict] = None) -> None:
        """
        Log a user interaction event.
        
        Args:
            user_id: User who interacted
            movie_id: Movie that was interacted with
            interaction_type: Type of interaction ("click", "watch", "rate", etc.)
            model: Which model recommended this (if known)
            session_id: Session identifier to link with recommendation
            metadata: Additional metadata (e.g., watch_duration, rating_value)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'interaction',
            'user_id': user_id,
            'movie_id': movie_id,
            'interaction_type': interaction_type,
            'model': model,
            'session_id': session_id
        }
        
        if metadata:
            entry.update(metadata)
        
        with self.lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def log_click(self, user_id: str, movie_id: str, model: Optional[str] = None) -> None:
        """Log a click on a recommendation."""
        self.log_interaction(user_id, movie_id, 'click', model)
    
    def log_watch(self, user_id: str, movie_id: str, 
                  duration_minutes: float, model: Optional[str] = None) -> None:
        """Log a watch event."""
        self.log_interaction(
            user_id, movie_id, 'watch', model,
            metadata={'watch_duration_minutes': duration_minutes}
        )
    
    def log_rating(self, user_id: str, movie_id: str,
                   rating: float, model: Optional[str] = None) -> None:
        """Log a rating event."""
        self.log_interaction(
            user_id, movie_id, 'rate', model,
            metadata={'rating_value': rating}
        )


# Global tracker instance
interaction_tracker = InteractionTracker()


def simulate_user_interactions(recommendations_log: str = "ab_test_results.jsonl",
                              click_rate: float = 0.25,
                              watch_rate: float = 0.15) -> None:
    """
    Simulate user interactions for demonstration.
    
    In production, these would come from your frontend analytics.
    
    Args:
        recommendations_log: Log file with recommendations
        click_rate: Probability a recommendation is clicked
        watch_rate: Probability a clicked recommendation is watched
    """
    import random
    from datetime import timedelta
    
    print("Simulating user interactions...")
    print(f"Click rate: {click_rate * 100}%")
    print(f"Watch rate (of clicks): {watch_rate * 100}%")
    print()
    
    # Load recommendations
    recommendations = []
    if Path(recommendations_log).exists():
        with open(recommendations_log, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('success') and entry.get('recommendations'):
                        timestamp = datetime.fromisoformat(entry['timestamp'])
                        for i, movie_id in enumerate(entry['recommendations'][:20], 1):
                            recommendations.append({
                                'user_id': entry['user_id'],
                                'movie_id': movie_id,
                                'model': entry['model'],
                                'position': i,
                                'timestamp': timestamp
                            })
                except (json.JSONDecodeError, KeyError):
                    continue
    
    print(f"Found {len(recommendations)} recommendations")
    
    click_count = 0
    watch_count = 0
    
    # Simulate interactions
    for rec in recommendations:
        # Higher positions more likely to be clicked
        position_boost = 3.0 if rec['position'] <= 3 else 1.0
        click_prob = click_rate * position_boost
        
        # Simulate click
        if random.random() < click_prob:
            # Log click event
            interaction_tracker.log_click(
                user_id=rec['user_id'],
                movie_id=rec['movie_id'],
                model=rec['model']
            )
            click_count += 1
            
            # Simulate watch (if clicked)
            if random.random() < watch_rate:
                # Random watch duration
                watch_duration = random.uniform(20, 120)
                
                interaction_tracker.log_watch(
                    user_id=rec['user_id'],
                    movie_id=rec['movie_id'],
                    duration_minutes=watch_duration,
                    model=rec['model']
                )
                watch_count += 1
    
    print(f" Simulated {click_count} clicks ({click_count/len(recommendations)*100:.1f}%)")
    print(f" Simulated {watch_count} watches ({watch_count/len(recommendations)*100:.1f}%)")
    print()


if __name__ == "__main__":
    # Example: Simulate interactions
    if Path("ab_test_results.jsonl").exists():
        simulate_user_interactions()
        print(f" Interactions logged to: user_interactions.jsonl")
    else:
        print("No recommendation data found. Run experiment first.")