"""
A/B Test Simulation Script

Simulates user traffic to test the A/B testing infrastructure.
Useful for demonstrating the experimentation system before having two different models.
"""

import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import sys


class ABTestSimulator:
    """Simulate A/B test traffic to the recommendation API."""
    
    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url
        self.results = []
        
    def make_request(self, user_id: int, n_recs: int = 20) -> Dict:
        """Make a single recommendation request."""
        url = f"{self.base_url}/recommend/{user_id}"
        params = {'n': n_recs} if n_recs != 20 else {}
        
        start_time = time.time()
        try:
            response = requests.get(url, params=params, timeout=5)
            elapsed = time.time() - start_time
            
            return {
                'user_id': user_id,
                'status_code': response.status_code,
                'response_time': elapsed,
                'success': response.status_code == 200,
                'n_recommendations': len(response.text.split(',')) if response.status_code == 200 else 0
            }
        except Exception as e:
            return {
                'user_id': user_id,
                'status_code': 0,
                'response_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'n_recommendations': 0
            }
    
    def simulate_traffic(self, 
                        n_requests: int = 100,
                        user_id_range: tuple = (1, 10000),
                        parallel: bool = True,
                        max_workers: int = 10) -> None:
        """
        Simulate user traffic.
        
        Args:
            n_requests: Total number of requests to make
            user_id_range: Range of user IDs to sample from
            parallel: Whether to make requests in parallel
            max_workers: Number of parallel workers
        """
        print(f"🚀 Starting A/B test simulation...")
        print(f"   Target: {n_requests} requests")
        print(f"   User ID range: {user_id_range[0]} - {user_id_range[1]}")
        print(f"   Parallel: {parallel} (workers: {max_workers if parallel else 'N/A'})")
        print()
        
        # Generate user IDs
        user_ids = [random.randint(*user_id_range) for _ in range(n_requests)]
        
        start_time = time.time()
        
        if parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.make_request, uid) for uid in user_ids]
                
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    self.results.append(result)
                    
                    if i % 10 == 0:
                        print(f"   Progress: {i}/{n_requests} requests completed", end='\r')
        else:
            # Sequential execution
            for i, uid in enumerate(user_ids, 1):
                result = self.make_request(uid)
                self.results.append(result)
                
                if i % 10 == 0:
                    print(f"   Progress: {i}/{n_requests} requests completed", end='\r')
                
                # Small delay to avoid overwhelming server
                time.sleep(0.01)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n\n✓ Simulation complete!")
        print(f"   Total time: {elapsed_time:.2f}s")
        print(f"   Requests/sec: {n_requests/elapsed_time:.2f}")
        print()
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        if not self.results:
            print("No results to summarize")
            return
        
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        print("=" * 70)
        print("SIMULATION SUMMARY")
        print("=" * 70)
        print(f"Total requests:     {len(self.results)}")
        print(f"Successful:         {len(successful)} ({len(successful)/len(self.results)*100:.1f}%)")
        print(f"Failed:             {len(failed)} ({len(failed)/len(self.results)*100:.1f}%)")
        print()
        
        if successful:
            response_times = [r['response_time'] for r in successful]
            print(f"Response times:")
            print(f"  Mean:             {sum(response_times)/len(response_times)*1000:.2f} ms")
            print(f"  Median:           {sorted(response_times)[len(response_times)//2]*1000:.2f} ms")
            print(f"  Min:              {min(response_times)*1000:.2f} ms")
            print(f"  Max:              {max(response_times)*1000:.2f} ms")
            print()
        
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Check API stats:  curl http://localhost:8082/experiment/stats")
        print("  2. Run analysis:     python -m ml_pipeline.serve.ab_test_analyzer")
        print("  3. View logs:        tail -f ab_test_results.jsonl")
        print()
    
    def check_api_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ API is healthy")
                print(f"  Model A loaded: {data.get('model_a_loaded', 'unknown')}")
                print(f"  Model B loaded: {data.get('model_b_loaded', 'unknown')}")
                return True
            else:
                print(f"✗ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot reach API: {e}")
            return False


def main():
    """Main entry point for simulation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate A/B test traffic')
    parser.add_argument('--url', default='http://localhost:8082',
                       help='Base URL of the API (default: http://localhost:8082)')
    parser.add_argument('--requests', type=int, default=100,
                       help='Number of requests to make (default: 100)')
    parser.add_argument('--min-user', type=int, default=1,
                       help='Minimum user ID (default: 1)')
    parser.add_argument('--max-user', type=int, default=10000,
                       help='Maximum user ID (default: 10000)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run requests sequentially instead of parallel')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    
    args = parser.parse_args()
    
    simulator = ABTestSimulator(base_url=args.url)
    
    # Check API health first
    print("Checking API health...")
    if not simulator.check_api_health():
        print("\nCannot proceed - API is not responding")
        print("Make sure the Flask app is running:")
        print("  python -m ml_pipeline.serve.app")
        sys.exit(1)
    
    print()
    
    # Run simulation
    simulator.simulate_traffic(
        n_requests=args.requests,
        user_id_range=(args.min_user, args.max_user),
        parallel=not args.sequential,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()