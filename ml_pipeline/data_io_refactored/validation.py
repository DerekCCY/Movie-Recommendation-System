from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class DataQualityReport:
    """Container for data quality check results."""
    entity_type: str  # 'movie' or 'user'
    entity_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    timestamp: str

    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        report = [f"{status} - {self.entity_type} {self.entity_id}"]
        if self.errors:
            report.append(f"  Errors: {', '.join(self.errors)}")
        if self.warnings:
            report.append(f"  Warnings: {', '.join(self.warnings)}")
        return "\n".join(report)


class DataQualityValidator:
    """Validates movie and user data from API responses."""

    MOVIE_REQUIRED_FIELDS = ["id", "title"]
    MOVIE_OPTIONAL_FIELDS = ["genres", "year", "rating", "director", "runtime"]

    USER_REQUIRED_FIELDS = ["id"]
    USER_OPTIONAL_FIELDS = ["age", "gender", "occupation", "zip_code"]

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    # ------------------ Movie ------------------
    def validate_movie(self, movie_data: Dict | None, movie_id: str) -> DataQualityReport:
        errors: List[str] = []
        warnings: List[str] = []

        if movie_data is None:
            errors.append("No data returned from API")
            return DataQualityReport("movie", movie_id, False, errors, warnings, datetime.now().isoformat())

        for field in self.MOVIE_REQUIRED_FIELDS:
            if field not in movie_data:
                errors.append(f"Missing required field: {field}")
            elif movie_data[field] in (None, ""):
                errors.append(f"Required field '{field}' is null or empty")

        for field in self.MOVIE_OPTIONAL_FIELDS:
            if field not in movie_data or movie_data[field] is None:
                warnings.append(f"Missing optional field: {field}")

        if (y := movie_data.get("year")) is not None:
            try:
                year = int(y)
                current_year = datetime.now().year
                if year < 1888:
                    errors.append(f"Invalid year: {year} (before 1888)")
                elif year > current_year + 2:
                    warnings.append(f"Year {year} is in the future")
            except (ValueError, TypeError):
                errors.append(f"Year must be numeric, got: {y}")

        if (r := movie_data.get("rating")) is not None:
            try:
                rating = float(r)
                if not (0 <= rating <= 10):
                    errors.append(f"Rating {rating} outside valid range [0, 10]")
            except (ValueError, TypeError):
                errors.append(f"Rating must be numeric, got: {r}")

        if (rt := movie_data.get("runtime")) is not None:
            try:
                runtime = int(rt)
                if runtime < 1:
                    errors.append(f"Invalid runtime: {runtime} minutes")
                elif runtime < 30:
                    warnings.append(f"Very short runtime: {runtime} minutes")
                elif runtime > 600:
                    warnings.append(f"Very long runtime: {runtime} minutes")
            except (ValueError, TypeError):
                errors.append(f"Runtime must be numeric, got: {rt}")

        genres = movie_data.get("genres")
        if genres is not None and not isinstance(genres, (list, str)):
            errors.append(f"Genres must be list or string, got: {type(genres)}")
        elif isinstance(genres, list) and len(genres) == 0:
            warnings.append("Empty genres list")

        if (mid := movie_data.get("id")) is not None and str(mid) != str(movie_id):
            errors.append(f"ID mismatch: requested {movie_id}, got {mid}")

        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0

        return DataQualityReport("movie", movie_id, is_valid, errors, warnings, datetime.now().isoformat())

    # ------------------ User ------------------
    def validate_user(self, user_data: Dict | None, user_id: str) -> DataQualityReport:
        errors: List[str] = []
        warnings: List[str] = []

        if user_data is None:
            errors.append("No data returned from API")
            return DataQualityReport("user", user_id, False, errors, warnings, datetime.now().isoformat())

        for field in self.USER_REQUIRED_FIELDS:
            if field not in user_data:
                errors.append(f"Missing required field: {field}")
            elif user_data[field] in (None, ""):
                errors.append(f"Required field '{field}' is null or empty")

        for field in self.USER_OPTIONAL_FIELDS:
            if field not in user_data or user_data[field] is None:
                warnings.append(f"Missing optional field: {field}")

        age = user_data.get("age")
        if age is not None:
            try:
                age_val = int(age)
                if age_val < 0:
                    errors.append(f"Invalid age: {age_val}")
                elif age_val < 13:
                    warnings.append(f"User age {age_val} below typical minimum")
                elif age_val > 120:
                    warnings.append(f"User age {age_val} seems unrealistic")
            except (ValueError, TypeError):
                errors.append(f"Age must be numeric, got: {age}")

        gender = user_data.get("gender")
        if gender is not None and gender != "":
            valid_genders = ["M", "F", "Male", "Female", "Other", "Non-binary"]
            if gender not in valid_genders:
                warnings.append(f"Unexpected gender value: {gender}")

        zip_code = user_data.get("zip_code")
        if zip_code is not None:
            z = str(zip_code)
            if not (len(z) == 5 or len(z) == 10):
                warnings.append(f"Unusual zip code format: {z}")

        uid = user_data.get("id")
        if uid is not None and str(uid) != str(user_id):
            errors.append(f"ID mismatch: requested {user_id}, got {uid}")

        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0

        return DataQualityReport("user", user_id, is_valid, errors, warnings, datetime.now().isoformat())