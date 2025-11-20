"""
Data I/O Module with Data Quality Checks

Handles data loading from various sources:
- Kafka event stream ingestion
- Parquet file loading
- External API calls (movie/user metadata)
- Caching for API responses
- Data quality validation
"""

import pandas as pd
import json
import time
import os
import requests
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


def load_interactions_from_parquet(path: str) -> pd.DataFrame:
    """
    Load user-movie interactions from a Parquet file.

    Args:
        path: Path to the parquet file containing interactions

    Returns:
        DataFrame with columns: user_id, movie_id, rating (optional), watch_minutes (optional)

    Example:
        >>> df = load_interactions_from_parquet("data/gold/interactions.parquet")
        >>> print(df.columns)
        Index(['user_id', 'movie_id', 'rating', 'watch_minutes'])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Interactions file not found: {path}")

    df = pd.read_parquet(path)
    return df


def load_kafka_events(config: Dict, num_messages: Optional[int] = None) -> pd.DataFrame:
    """
    Consume events from Kafka topic and return as DataFrame.

    Args:
        config: Kafka configuration dict (bootstrap_servers, topic, group_id, etc.)
        num_messages: Optional limit on number of messages to consume

    Returns:
        DataFrame with raw event logs

    Note:
        This is a wrapper around existing src/ingestion/kafka_consumer.py logic
        For M2, we focus on batch processing. Live Kafka ingestion is out of scope.
    """
    # TODO: Implement if needed for M2
    # For now, teams should use pre-collected data
    raise NotImplementedError("Kafka ingestion not required for M2 - use batch data files")


@dataclass
class DataQualityReport:
    """Container for data quality check results."""
    entity_type: str  # 'movie' or 'user'
    entity_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    timestamp: str
    
    def __str__(self):
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        report = [f"{status} - {self.entity_type} {self.entity_id}"]
        if self.errors:
            report.append(f"  Errors: {', '.join(self.errors)}")
        if self.warnings:
            report.append(f"  Warnings: {', '.join(self.warnings)}")
        return "\n".join(report)


class DataQualityValidator:
    """
    Validates movie and user data from API responses.
    """
    
    # Expected fields for each entity type
    MOVIE_REQUIRED_FIELDS = ['id', 'title']
    MOVIE_OPTIONAL_FIELDS = ['genres', 'year', 'rating', 'director', 'runtime']
    
    USER_REQUIRED_FIELDS = ['id']
    USER_OPTIONAL_FIELDS = ['age', 'gender', 'occupation', 'zip_code']
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
    
    def validate_movie(self, movie_data: Optional[Dict], movie_id: str) -> DataQualityReport:
        """
        Validate movie data quality.
        
        Args:
            movie_data: Movie metadata dict from API
            movie_id: Movie identifier for reporting
            
        Returns:
            DataQualityReport with validation results
        """
        errors = []
        warnings = []
        
        # Check if data exists
        if movie_data is None:
            errors.append("No data returned from API")
            return DataQualityReport(
                entity_type='movie',
                entity_id=movie_id,
                is_valid=False,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now().isoformat()
            )
        
        # Check required fields
        for field in self.MOVIE_REQUIRED_FIELDS:
            if field not in movie_data:
                errors.append(f"Missing required field: {field}")
            elif movie_data[field] is None or movie_data[field] == "":
                errors.append(f"Required field '{field}' is null or empty")
        
        # Check optional fields (warnings only)
        for field in self.MOVIE_OPTIONAL_FIELDS:
            if field not in movie_data or movie_data[field] is None:
                warnings.append(f"Missing optional field: {field}")
        
        # Validate data types and ranges
        if 'year' in movie_data and movie_data['year'] is not None:
            try:
                year = int(movie_data['year'])
                current_year = datetime.now().year
                if year < 1888:  # First movie ever made
                    errors.append(f"Invalid year: {year} (before 1888)")
                elif year > current_year + 2:  # Allow some future releases
                    warnings.append(f"Year {year} is in the future")
            except (ValueError, TypeError):
                errors.append(f"Year must be numeric, got: {movie_data['year']}")
        
        if 'rating' in movie_data and movie_data['rating'] is not None:
            try:
                rating = float(movie_data['rating'])
                if not (0 <= rating <= 10):
                    errors.append(f"Rating {rating} outside valid range [0, 10]")
            except (ValueError, TypeError):
                errors.append(f"Rating must be numeric, got: {movie_data['rating']}")
        
        if 'runtime' in movie_data and movie_data['runtime'] is not None:
            try:
                runtime = int(movie_data['runtime'])
                if runtime < 1:
                    errors.append(f"Invalid runtime: {runtime} minutes")
                elif runtime < 30:
                    warnings.append(f"Very short runtime: {runtime} minutes")
                elif runtime > 600:
                    warnings.append(f"Very long runtime: {runtime} minutes")
            except (ValueError, TypeError):
                errors.append(f"Runtime must be numeric, got: {movie_data['runtime']}")
        
        # Validate genres format
        if 'genres' in movie_data and movie_data['genres'] is not None:
            if not isinstance(movie_data['genres'], (list, str)):
                errors.append(f"Genres must be list or string, got: {type(movie_data['genres'])}")
            elif isinstance(movie_data['genres'], list) and len(movie_data['genres']) == 0:
                warnings.append("Empty genres list")
        
        # Check for duplicate ID mismatch
        if 'id' in movie_data and str(movie_data['id']) != str(movie_id):
            errors.append(f"ID mismatch: requested {movie_id}, got {movie_data['id']}")
        
        # Determine validity
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        return DataQualityReport(
            entity_type='movie',
            entity_id=movie_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now().isoformat()
        )
    
    def validate_user(self, user_data: Optional[Dict], user_id: str) -> DataQualityReport:
        """
        Validate user data quality.
        
        Args:
            user_data: User metadata dict from API
            user_id: User identifier for reporting
            
        Returns:
            DataQualityReport with validation results
        """
        errors = []
        warnings = []
        
        # Check if data exists
        if user_data is None:
            errors.append("No data returned from API")
            return DataQualityReport(
                entity_type='user',
                entity_id=user_id,
                is_valid=False,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now().isoformat()
            )
        
        # Check required fields
        for field in self.USER_REQUIRED_FIELDS:
            if field not in user_data:
                errors.append(f"Missing required field: {field}")
            elif user_data[field] is None or user_data[field] == "":
                errors.append(f"Required field '{field}' is null or empty")
        
        # Check optional fields
        for field in self.USER_OPTIONAL_FIELDS:
            if field not in user_data or user_data[field] is None:
                warnings.append(f"Missing optional field: {field}")
        
        # Validate age
        if 'age' in user_data and user_data['age'] is not None:
            try:
                age = int(user_data['age'])
                if age < 0:
                    errors.append(f"Invalid age: {age}")
                elif age < 13:
                    warnings.append(f"User age {age} below typical minimum")
                elif age > 120:
                    warnings.append(f"User age {age} seems unrealistic")
            except (ValueError, TypeError):
                errors.append(f"Age must be numeric, got: {user_data['age']}")
        
        # Validate gender
        if 'gender' in user_data and user_data['gender'] is not None:
            valid_genders = ['M', 'F', 'Male', 'Female', 'Other', 'Non-binary']
            if user_data['gender'] not in valid_genders and user_data['gender'] != "":
                warnings.append(f"Unexpected gender value: {user_data['gender']}")
        
        # Validate zip code format (US)
        if 'zip_code' in user_data and user_data['zip_code'] is not None:
            zip_code = str(user_data['zip_code'])
            if not (len(zip_code) == 5 or len(zip_code) == 10):  # 12345 or 12345-6789
                warnings.append(f"Unusual zip code format: {zip_code}")
        
        # Check for ID mismatch
        if 'id' in user_data and str(user_data['id']) != str(user_id):
            errors.append(f"ID mismatch: requested {user_id}, got {user_data['id']}")
        
        # Determine validity
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        return DataQualityReport(
            entity_type='user',
            entity_id=user_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now().isoformat()
        )


class MovieAPIClient:
    """
    Client for fetching movie and user metadata from external API.
    Includes caching, rate limiting, and data quality validation.

    Enhanced from model/model2/model2.2.py
    """

    def __init__(self, base_url: str = "http://128.2.220.241:8080",
                cache_path: Optional[str] = None,
                request_delay: float = 0.1,
                enable_validation: bool = True,
                strict_validation: bool = False):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the movie API
            cache_path: Path to JSON file for caching API responses
            request_delay: Delay between requests in seconds (rate limiting)
            enable_validation: Whether to run data quality checks
            strict_validation: If True, warnings are treated as errors
        """
        self.base_url = base_url
        self.cache_path = cache_path
        self.request_delay = request_delay
        self.cache = {}
        self.session = requests.Session()
        self.enable_validation = enable_validation
        self.validator = DataQualityValidator(strict_mode=strict_validation)
        
        # Track quality metrics
        self.quality_reports: List[DataQualityReport] = []

        if cache_path and os.path.exists(cache_path):
            self.load_cache(cache_path)

    def get_movie(self, movie_id: str, validate: Optional[bool] = None) -> Optional[Dict]:
        """
        Fetch movie metadata by ID with optional validation.

        Args:
            movie_id: Movie identifier
            validate: Override instance-level validation setting

        Returns:
            Dict with movie metadata (title, genres, year, etc.) or None if error
        """
        if movie_id in self.cache:
            return self.cache[movie_id]

        try:
            time.sleep(self.request_delay)
            response = self.session.get(f"{self.base_url}/movie/{movie_id}", timeout=10)
            response.raise_for_status()
            if response.status_code != 200:
                return None 
            data = response.json()
            
            # Validate if enabled
            should_validate = validate if validate is not None else self.enable_validation
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

    def get_user(self, user_id: str, validate: Optional[bool] = None) -> Optional[Dict]:
        """
        Fetch user metadata by ID with optional validation.

        Args:
            user_id: User identifier
            validate: Override instance-level validation setting

        Returns:
            Dict with user metadata or None if error
        """
        key = f"user_{user_id}"
        if key in self.cache:
            return self.cache[key]

        try:
            time.sleep(self.request_delay)
            response = self.session.get(f"{self.base_url}/user/{user_id}", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Validate if enabled
            should_validate = validate if validate is not None else self.enable_validation
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

    def get_movies_batch(self, movie_ids: List[str], validate: Optional[bool] = None) -> Dict[str, Dict]:
        """
        Fetch metadata for multiple movies with optional validation.

        Args:
            movie_ids: List of movie identifiers
            validate: Override instance-level validation setting

        Returns:
            Dict mapping movie_id -> metadata dict
        """
        results = {}
        for movie_id in movie_ids:
            movie_data = self.get_movie(movie_id, validate=validate)
            if movie_data:
                results[movie_id] = movie_data
        return results

    def get_quality_summary(self) -> Dict:
        """
        Get summary statistics of data quality checks.
        
        Returns:
            Dict with quality metrics
        """
        if not self.quality_reports:
            return {"message": "No quality reports available"}
        
        total = len(self.quality_reports)
        valid = sum(1 for r in self.quality_reports if r.is_valid)
        invalid = total - valid
        
        movies = [r for r in self.quality_reports if r.entity_type == 'movie']
        users = [r for r in self.quality_reports if r.entity_type == 'user']
        
        all_errors = [e for r in self.quality_reports for e in r.errors]
        all_warnings = [w for r in self.quality_reports for w in r.warnings]
        
        return {
            "total_checks": total,
            "valid": valid,
            "invalid": invalid,
            "validity_rate": f"{(valid/total*100):.1f}%",
            "movies_checked": len(movies),
            "users_checked": len(users),
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
            "common_errors": self._get_common_issues(all_errors),
            "common_warnings": self._get_common_issues(all_warnings)
        }
    
    def _get_common_issues(self, issues: List[str], top_n: int = 3) -> List[Tuple[str, int]]:
        """Get most common issues from list."""
        from collections import Counter
        if not issues:
            return []
        return Counter(issues).most_common(top_n)
    
    def export_quality_report(self, path: str) -> None:
        """
        Export quality reports to JSON file.
        
        Args:
            path: Path to save quality report
        """
        report_data = {
            "summary": self.get_quality_summary(),
            "reports": [
                {
                    "entity_type": r.entity_type,
                    "entity_id": r.entity_id,
                    "is_valid": r.is_valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "timestamp": r.timestamp
                }
                for r in self.quality_reports
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"✓ Quality report exported to {path}")

    def save_cache(self, path: Optional[str] = None) -> None:
        """
        Save API response cache to JSON file.

        Args:
            path: Path to save cache (uses self.cache_path if not provided)
        """
        save_path = path or self.cache_path
        if not save_path:
            raise ValueError("No cache path specified")

        try:
            with open(save_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_cache(self, path: str) -> None:
        """
        Load API response cache from JSON file.

        Args:
            path: Path to cached JSON file
        """
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.cache = {}


# Example usage
if __name__ == "__main__":
    # Initialize client with validation enabled
    client = MovieAPIClient(
        cache_path="cache/movies.json",
        enable_validation=True,
        strict_validation=False  # Warnings won't fail validation
    )
    
    # Fetch some movies
    movie_ids = ["1", "2", "999999"]  # Last one might not exist
    movies = client.get_movies_batch(movie_ids)
    
    # Get quality summary
    summary = client.get_quality_summary()
    print("\n" + "="*50)
    print("DATA QUALITY SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export detailed report
    client.export_quality_report("quality_report.json")
    
    # Save cache
    client.save_cache()