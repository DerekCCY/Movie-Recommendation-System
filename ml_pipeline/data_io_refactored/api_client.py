import json
import os
import time
from typing import Dict, List, Optional, Tuple

import requests

from .validation import DataQualityReport, DataQualityValidator


class MovieAPIClient:
    """
    Client for fetching movie and user metadata from external API.
    Includes caching, rate limiting, and data quality validation.
    """

    def __init__(
        self,
        base_url: str = "http://128.2.220.241:8080",
        cache_path: Optional[str] = None,
        request_delay: float = 0.1,
        enable_validation: bool = True,
        strict_validation: bool = False,
    ):
        self.base_url = base_url
        self.cache_path = cache_path
        self.request_delay = request_delay
        self.cache: Dict[str, Dict] = {}
        self.session = requests.Session()
        self.enable_validation = enable_validation
        self.validator = DataQualityValidator(strict_mode=strict_validation)
        self.quality_reports: List[DataQualityReport] = []

        if cache_path and os.path.exists(cache_path):
            self.load_cache(cache_path)

    # ------------------ Movie ------------------
    def get_movie(self, movie_id: str, validate: Optional[bool] = None) -> Optional[Dict]:
        if movie_id in self.cache:
            return self.cache[movie_id]
        try:
            time.sleep(self.request_delay)
            response = self.session.get(f"{self.base_url}/movie/{movie_id}", timeout=10)
            response.raise_for_status()
            if response.status_code != 200:
                return None
            data = response.json()
            should_validate = self.enable_validation if validate is None else validate
            if should_validate:
                report = self.validator.validate_movie(data, movie_id)
                self.quality_reports.append(report)
                if not report.is_valid:
                    print(f"⚠️  Data quality issues for movie {movie_id}:")
                    for error in report.errors:
                        print(f"   - {error}")
            self.cache[movie_id] = data
            return data
        except Exception as e:
            print(f"Error fetching movie {movie_id}: {e}")
            return None

    # ------------------ User ------------------
    def get_user(self, user_id: str, validate: Optional[bool] = None) -> Optional[Dict]:
        key = f"user_{user_id}"
        if key in self.cache:
            return self.cache[key]
        try:
            time.sleep(self.request_delay)
            response = self.session.get(f"{self.base_url}/user/{user_id}", timeout=10)
            response.raise_for_status()
            data = response.json()
            should_validate = self.enable_validation if validate is None else validate
            if should_validate:
                report = self.validator.validate_user(data, user_id)
                self.quality_reports.append(report)
                if not report.is_valid:
                    print(f"⚠️  Data quality issues for user {user_id}:")
                    for error in report.errors:
                        print(f"   - {error}")
            self.cache[key] = data
            return data
        except Exception as e:
            print(f"Error fetching user {user_id}: {e}")
            return None

    # ------------------ Batch helpers ------------------
    def get_movies_batch(self, movie_ids: List[str], validate: Optional[bool] = None) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for movie_id in movie_ids:
            movie_data = self.get_movie(movie_id, validate=validate)
            if movie_data:
                results[movie_id] = movie_data
        return results

    # ------------------ Quality report helpers ------------------
    def get_quality_summary(self) -> Dict:
        if not self.quality_reports:
            return {"message": "No quality reports available"}
        total = len(self.quality_reports)
        valid = sum(1 for r in self.quality_reports if r.is_valid)
        invalid = total - valid
        movies = [r for r in self.quality_reports if r.entity_type == "movie"]
        users = [r for r in self.quality_reports if r.entity_type == "user"]
        all_errors = [e for r in self.quality_reports for e in r.errors]
        all_warnings = [w for r in self.quality_reports for w in r.warnings]
        from collections import Counter
        def top(items, n=3):
            return Counter(items).most_common(n) if items else []
        return {
            "total_checks": total,
            "valid": valid,
            "invalid": invalid,
            "validity_rate": f"{(valid/total*100):.1f}%",
            "movies_checked": len(movies),
            "users_checked": len(users),
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "common_errors": top(all_errors),
            "common_warnings": top(all_warnings),
        }

    def export_quality_report(self, path: str) -> None:
        report_data = {
            "summary": self.get_quality_summary(),
            "reports": [
                {
                    "entity_type": r.entity_type,
                    "entity_id": r.entity_id,
                    "is_valid": r.is_valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "timestamp": r.timestamp,
                }
                for r in self.quality_reports
            ],
        }
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"✓ Quality report exported to {path}")

    # ------------------ Cache helpers ------------------
    def save_cache(self, path: Optional[str] = None) -> None:
        save_path = path or self.cache_path
        if not save_path:
            raise ValueError("No cache path specified")
        with open(save_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def load_cache(self, path: str) -> None:
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.cache = {}