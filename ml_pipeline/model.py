"""
SVD Recommendation Model

Implementation of Singular Value Decomposition (SVD) with bias terms
for collaborative filtering movie recommendations.

Extracted from:
- deployment/src/model_class.py (M1 with cold start)
- model/model2/model2.2.py (training logic)
"""

import numpy as np
from scipy.sparse.linalg import svds
from typing import List, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
from collections import Counter
import pandas as pd
import time

class ImprovedSVDRecommendationModel:
    """
    SVD-based collaborative filtering model with bias terms.

    Learns latent user and item factors plus global/user/item biases to predict ratings.
    Prediction formula: r_ui = μ + b_u + b_i + q_i^T * p_u

    Where:
    - μ = global mean rating
    - b_u = user bias
    - b_i = item bias
    - q_i = item latent factors
    - p_u = user latent factors
    """

    def __init__(self, n_factors: int = 100, regularization: float = 0.01):
        """
        Initialize SVD model.

        Args:
            n_factors: Number of latent factors to learn
            regularization: L2 regularization parameter
        """
        self.n_factors = n_factors
        self.regularization = regularization

        # Learned parameters (set during fit())
        self.user_factors = None  # Shape: (n_users, n_factors)
        self.item_factors = None  # Shape: (n_items, n_factors)
        self.global_mean = None  # Scalar
        self.user_bias = None  # Shape: (n_users,)
        self.item_bias = None  # Shape: (n_items,)

        # Mappings
        self.user_mapping = {}  # {user_id: index}
        self.item_mapping = {}  # {movie_id: index}
        self.reverse_user_mapping = {}  # {index: user_id}
        self.reverse_item_mapping = {}  # {index: movie_id}

        # Metadata
        self.all_movie_ids = []
        self.user_ids = []
        self.popular_movies = []  # For cold start
        self.n_users = 0
        self.n_items = 0

    def fit(self, ratings_matrix, mappings: dict, global_stats: dict,
            popular_movies: List[str]) -> None:
        """
        Train the SVD model using SVD decomposition.

        Args:
            ratings_matrix: Sparse user-item rating matrix (scipy.sparse.csr_matrix)
            mappings: Dict with user/item mappings from features.py
            global_stats: Dict with global_mean, user_biases, item_biases
            popular_movies: List of popular movie IDs for cold start

        Note:
            Extracted from model/model2/model2.2.py lines 310-330
        """
        # Store mappings
        self.user_mapping = mappings['user_mapping']
        self.item_mapping = mappings['item_mapping']
        self.reverse_user_mapping = mappings['reverse_user_mapping']
        self.reverse_item_mapping = mappings['reverse_item_mapping']
        self.n_users = mappings['n_users']
        self.n_items = mappings['n_items']

        # Store metadata
        self.user_ids = list(self.user_mapping.keys())
        self.all_movie_ids = list(self.item_mapping.keys())
        self.popular_movies = popular_movies

        # Store global statistics
        self.global_mean = global_stats['global_mean']
        self.user_bias = global_stats['user_biases']
        self.item_bias = global_stats['item_biases']

        # Perform SVD decomposition
        k = min(self.n_factors, min(self.n_users, self.n_items) - 1)
        U, sigma, Vt = svds(ratings_matrix, k=k)

        # Convert to user and item factors
        sigma = np.diag(sigma)
        self.user_factors = np.dot(U, sigma).astype(np.float32)
        self.item_factors = Vt.T.astype(np.float32)

    def predict_rating(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.

        Args:
            user_id: User identifier
            movie_id: Movie identifier

        Returns:
            Predicted rating (clipped to [1, 5])

        Note:
            Extracted from deployment/src/model_class.py lines 33-50
        """
        user_id, movie_id = str(user_id), str(movie_id)

        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            return self.global_mean

        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[movie_id]

        # Prediction with bias terms
        prediction = (np.dot(self.user_factors[user_idx], self.item_factors[item_idx]) +
                     self.global_mean +
                     self.user_bias[user_idx] +
                     self.item_bias[item_idx])

        # Clip to valid range
        return np.clip(prediction, 1.0, 5.0)

    def predict(self, user_id: str, n_recommendations: int = 20) -> List[str]:
        """
        Generate movie recommendations for a user.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations to return

        Returns:
            List of movie_ids sorted by predicted rating (descending)

        Note:
            For unknown users, uses hash-based diversified cold start strategy
            Extracted from deployment/src/model_class.py lines 52-76
        """
        user_id = str(user_id)

        if user_id not in self.user_mapping:
            # Cold start: diversify recommendations based on user_id hash
            return self._cold_start_recommendations(user_id, n_recommendations)

        try:
            user_idx = self.user_mapping[user_id]

            # Calculate scores for all items with bias
            scores = (np.dot(self.item_factors, self.user_factors[user_idx]) +
                    self.global_mean +
                    self.user_bias[user_idx] +
                    self.item_bias)

            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            recommendations = [self.reverse_item_mapping[idx] for idx in top_indices]

            return recommendations

        except Exception as e:
            print(f"Error in prediction for user {user_id}: {e}")
            return self._cold_start_recommendations(user_id, n_recommendations)

    def _cold_start_recommendations(self, user_id: str,
                                   n_recommendations: int = 20) -> List[str]:
        """
        Generate diversified recommendations for unknown users.

        Uses hash of user_id to deterministically sample from top popular movies.
        Ensures different users get different recommendations.

        Args:
            user_id: User identifier (unknown to model)
            n_recommendations: Number of recommendations to return

        Returns:
            List of movie_ids sampled from popular movies

        Note:
            Extracted from deployment/src/model_class.py lines 78-96 (M1 improvement)
        """
        if not self.popular_movies:
            return []

        # Use hash of user_id to pick random subset from top 50 popular movies
        # This ensures different users get different (but consistent) recommendations
        user_hash = hash(user_id)
        np.random.seed(user_hash % (2**31))  # Use modulo to keep within valid range

        # Get top 50 popular movies and randomly sample from them
        top_movies = self.popular_movies[:50] if len(self.popular_movies) >= 50 else self.popular_movies

        # Randomly sample n_recommendations from top movies
        if len(top_movies) <= n_recommendations:
            return top_movies

        selected = np.random.choice(len(top_movies), size=n_recommendations, replace=False)
        return [top_movies[i] for i in selected]

    def get_model_info(self) -> dict:
        """
        Get model metadata and statistics.

        Returns:
            Dict with n_users, n_items, n_factors, etc.
        """
        return {
            'algorithm': 'Improved SVD with Bias Terms',
            'n_factors': self.n_factors,
            'regularization': self.regularization,
            'total_movies': len(self.all_movie_ids),
            'total_users': len(self.user_ids),
            'matrix_size': f"{self.n_users}×{self.n_items}" if hasattr(self, 'n_users') else "Unknown"
        }
        
        
"""
Neural Hybrid Recommendation Model
--------------------------------------------
Combines embeddings for users and movies with content metadata features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralHybridRecommender(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 n_metadata_features: int, embed_dim: int = 64):
        super().__init__()
        # Embeddings for IDs
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        # MLP input = user_emb + item_emb + metadata
        self.fc1 = nn.Linear(embed_dim * 2 + n_metadata_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_idx, item_idx, metadata_vec):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i, metadata_vec], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)

"""
Personalized Hybrid Recommendation Model
--------------------------------------------
Combines Collaborative Filtering with Content-Based features (genres, popularity, quality)
and personalized user preferences.

Author: Nora Ngo
"""

class PersonalizedHybridRecommender:
    """
    Hybrid recommender combining:
    1. Collaborative Filtering (user-item interactions)
    2. Content-Based Filtering (genres, popularity, ratings)
    3. Personalized Genre Preferences (user's genre history)
    """

    def __init__(self, n_factors=100, learning_rate=0.015, regularization=0.005,
                 content_weight=0.25):
        # Collaborative filtering params
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.regularization = regularization
        self.content_weight = content_weight  # Weight for content features

        # CF components
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None

        # Content-based components
        self.genre_mlb = MultiLabelBinarizer()
        self.item_genre_matrix = None
        self.item_popularity = {}
        self.item_quality = {}  # vote_average

        # Personalized user preferences
        self.user_genre_preferences = {}  # User's favorite genres
        self.user_quality_preference = {}  # User's preferred quality level
        self.user_popularity_preference = {}  # User's popularity preference

        # Mappings
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}

        # User history
        self.user_watched_items = {}
        self.user_ratings = {}

        # Metadata
        self.all_movie_ids = []
        self.user_ids = []
        self.n_users = 0
        self.n_items = 0
        self.training_history = []
        self.best_val_rmse = float('inf')
        self.patience_counter = 0

        # Statistics
        self.runtime_based_ratings = 0
        self.absolute_time_ratings = 0

    def _minutes_to_rating(self, minutes, runtime=None):
        """Convert watch minutes to rating, considering movie length"""
        if minutes <= 0:
            return 1

        if runtime and runtime > 0:
            self.runtime_based_ratings += 1
            pct_watched = minutes / runtime

            if pct_watched < 0.15:
                return 1
            elif pct_watched < 0.35:
                return 2
            elif pct_watched < 0.60:
                return 3
            elif pct_watched < 0.85:
                return 4
            else:
                return 5
        else:
            self.absolute_time_ratings += 1
            if minutes <= 10:
                return 2
            elif minutes <= 30:
                return 3
            elif minutes <= 60:
                return 4
            else:
                return 5

    def extract_genres(self, genres_str):
        """Extract genre list from string representation"""
        # Handle None or NaN
        if genres_str is None:
            return []

        try:
            # Check if it's already a list
            if isinstance(genres_str, list):
                return [str(g).strip() for g in genres_str if g]

            # Check if it's a numpy array
            if isinstance(genres_str, np.ndarray):
                return [str(g).strip() for g in genres_str.tolist() if g]

            # Handle pandas NA/NaN
            if pd.isna(genres_str):
                return []

            # Handle string representation
            if isinstance(genres_str, str):
                if genres_str == '' or genres_str == '[]':
                    return []
                # Remove brackets and quotes, split by comma
                genres_str = genres_str.strip('[]').replace("'", "").replace('"', '')
                genres = [g.strip() for g in genres_str.split(',') if g.strip()]
                return genres

            return []
        except Exception as e:
            # Fallback for any unexpected format
            return []

    def build_user_profiles(self, interactions_df):
        """Build personalized user profiles based on their interaction history"""
        print("\n=== Building User Profiles ===")

        # Use groupby for efficient iteration (100x faster than filtering)
        grouped = interactions_df.groupby('user_id')
        for user_id, user_data in grouped:

            # Extract user's genre preferences (weighted by ratings)
            genre_scores = Counter()
            quality_scores = []
            popularity_scores = []

            for _, row in user_data.iterrows():
                # Safely convert rating to float
                rating = row.get('rating', 0)
                try:
                    if pd.notna(rating):
                        rating = float(rating)
                    else:
                        rating = 0.0
                except (ValueError, TypeError):
                    rating = 0.0

                # Weight by rating (higher rated = stronger preference)
                weight = max(rating, 3.0)  # Minimum weight of 3

                # Genre preferences
                genres = self.extract_genres(row.get('genres', ''))
                for genre in genres:
                    genre_scores[genre] += weight

                # Quality preference - safely convert to float
                vote_avg = row.get('vote_average', None)
                try:
                    if vote_avg is not None and pd.notna(vote_avg):
                        vote_avg = float(vote_avg)
                        if vote_avg > 0:
                            quality_scores.append(vote_avg)
                except (ValueError, TypeError):
                    pass

                # Popularity preference - safely convert to float
                pop = row.get('popularity', None)
                try:
                    if pop is not None and pd.notna(pop):
                        pop = float(pop)
                        if pop > 0:
                            popularity_scores.append(pop)
                except (ValueError, TypeError):
                    pass

            # Store user's top genres
            if genre_scores:
                total_weight = sum(genre_scores.values())
                self.user_genre_preferences[str(user_id)] = {
                    genre: score / total_weight
                    for genre, score in genre_scores.most_common(10)
                }

            # Store user's preferred quality level
            if quality_scores:
                self.user_quality_preference[str(user_id)] = np.mean(quality_scores)

            # Store user's preferred popularity level
            if popularity_scores:
                self.user_popularity_preference[str(user_id)] = np.mean(popularity_scores)

        print(f"Built profiles for {len(self.user_genre_preferences)} users")

        # Show sample user profile
        if self.user_genre_preferences:
            sample_user = list(self.user_genre_preferences.keys())[0]
            print(f"\nSample User Profile (User {sample_user}):")
            print(f"  Top Genres: {dict(list(self.user_genre_preferences[sample_user].items())[:3])}")
            if sample_user in self.user_quality_preference:
                print(f"  Quality Preference: {self.user_quality_preference[sample_user]:.2f}")
            if sample_user in self.user_popularity_preference:
                print(f"  Popularity Preference: {self.user_popularity_preference[sample_user]:.2f}")

    def create_ratings_from_interactions(self, interactions_df, strategy='hybrid', min_interactions=3):
        """Convert interactions to ratings with runtime-aware conversion"""
        print(f"\n=== Processing Interactions (strategy: {strategy}) ===")

        self.runtime_based_ratings = 0
        self.absolute_time_ratings = 0

        ratings_data = []
        for _, row in interactions_df.iterrows():
            user_id = str(row['user_id'])
            movie_id = str(row['movie_id'])
            explicit_rating = row.get('rating')
            watch_minutes = row.get('watch_minutes', 0)
            runtime = row.get('runtime', None)

            rating = None
            if strategy == 'explicit_only' and pd.notna(explicit_rating):
                rating = float(explicit_rating)
            elif strategy == 'implicit_only' and watch_minutes > 0:
                rating = self._minutes_to_rating(watch_minutes, runtime)
            elif strategy == 'hybrid':
                if pd.notna(explicit_rating):
                    rating = float(explicit_rating)
                elif watch_minutes > 0:
                    rating = self._minutes_to_rating(watch_minutes, runtime)

            if rating is not None and 1 <= rating <= 5:
                ratings_data.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating
                })

        ratings_df = pd.DataFrame(ratings_data)
        ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

        print(f"Created {len(ratings_df)} ratings")
        print(f"  Runtime-based conversions: {self.runtime_based_ratings}")
        print(f"  Time-based fallbacks: {self.absolute_time_ratings}")

        ratings_df = self.smart_filtering(ratings_df, min_interactions)

        return ratings_df

    def smart_filtering(self, ratings_df, min_interactions=3):
        """Filter to keep quality data"""
        user_counts = ratings_df['user_id'].value_counts()
        movie_counts = ratings_df['movie_id'].value_counts()

        valid_users = user_counts[user_counts >= min_interactions].index
        valid_movies = movie_counts[movie_counts >= min_interactions].index

        filtered_df = ratings_df[
            (ratings_df['user_id'].isin(valid_users)) &
            (ratings_df['movie_id'].isin(valid_movies))
        ]

        print(f"After filtering: {len(filtered_df)} ratings, {filtered_df['user_id'].nunique()} users, {filtered_df['movie_id'].nunique()} movies")
        return filtered_df

    def prepare_content_features(self, interactions_df, train_df):
        """Prepare content-based features for all items"""
        print("\n=== Preparing Content Features ===")

        # Get unique movies from training data
        unique_movies = train_df['movie_id'].unique()

        # Create movie metadata lookup (one row per movie for efficiency)
        movie_metadata = interactions_df.drop_duplicates(subset='movie_id').set_index('movie_id')

        # Extract content features for each movie
        movie_features = {}
        all_genres = []

        for movie_id in unique_movies:
            movie_data = movie_metadata.loc[movie_id]

            # Extract genres
            genres = self.extract_genres(movie_data.get('genres', ''))
            all_genres.extend(genres)

            # Safely extract and convert numeric features
            try:
                popularity = float(movie_data.get('popularity', 0)) if pd.notna(movie_data.get('popularity')) else 0
            except (ValueError, TypeError):
                popularity = 0

            try:
                vote_average = float(movie_data.get('vote_average', 0)) if pd.notna(movie_data.get('vote_average')) else 0
            except (ValueError, TypeError):
                vote_average = 0

            try:
                vote_count = int(movie_data.get('vote_count', 0)) if pd.notna(movie_data.get('vote_count')) else 0
            except (ValueError, TypeError):
                vote_count = 0

            # Store features
            movie_features[movie_id] = {
                'genres': genres,
                'popularity': popularity,
                'vote_average': vote_average,
                'vote_count': vote_count
            }

        # Create genre matrix
        genre_lists = [movie_features[mid]['genres'] for mid in unique_movies]
        self.genre_mlb.fit(genre_lists)

        # Build item feature vectors
        self.item_features = {}
        for movie_id in unique_movies:
            features = movie_features[movie_id]

            # Genre encoding
            genre_vector = self.genre_mlb.transform([features['genres']])[0]

            self.item_features[movie_id] = {
                'genre_vector': genre_vector,
                'popularity': features['popularity'],
                'vote_average': features['vote_average'],
                'vote_count': features['vote_count']
            }

        # Store for later use
        self.item_popularity = {mid: f['popularity'] for mid, f in self.item_features.items()}
        self.item_quality = {mid: f['vote_average'] for mid, f in self.item_features.items()}

        print(f"Prepared features for {len(self.item_features)} movies")
        print(f"Total unique genres: {len(self.genre_mlb.classes_)}")
        print(f"Genres: {list(self.genre_mlb.classes_)[:10]}...")

    def prepare_vectorized_features(self):
        """Prepare vectorized matrices for fast batch prediction"""
        print("\n=== Preparing Vectorized Features for Fast Inference ===")

        # 1. Item genre matrix: (n_items, n_genres)
        n_genres = len(self.genre_mlb.classes_)
        self.item_genre_matrix = np.zeros((self.n_items, n_genres), dtype=np.float32)

        for movie_id, features in self.item_features.items():
            if movie_id in self.item_mapping:
                item_idx = self.item_mapping[movie_id]
                self.item_genre_matrix[item_idx] = features['genre_vector']

        # 2. Item quality vector: (n_items,)
        self.item_quality_vector = np.zeros(self.n_items, dtype=np.float32)
        for movie_id, features in self.item_features.items():
            if movie_id in self.item_mapping:
                item_idx = self.item_mapping[movie_id]
                self.item_quality_vector[item_idx] = features['vote_average']

        # 3. Item popularity vector: (n_items,)
        self.item_popularity_vector = np.zeros(self.n_items, dtype=np.float32)
        for movie_id, pop in self.item_popularity.items():
            if movie_id in self.item_mapping:
                item_idx = self.item_mapping[movie_id]
                self.item_popularity_vector[item_idx] = pop

        # 4. User genre preference matrix: (n_users, n_genres)
        self.user_genre_matrix = np.zeros((self.n_users, n_genres), dtype=np.float32)
        for user_id, genre_prefs in self.user_genre_preferences.items():
            if user_id in self.user_mapping:
                user_idx = self.user_mapping[user_id]
                for genre, score in genre_prefs.items():
                    genre_list = list(self.genre_mlb.classes_)
                    if genre in genre_list:
                        genre_idx = genre_list.index(genre)
                        self.user_genre_matrix[user_idx, genre_idx] = score

        # 5. User quality preference vector: (n_users,)
        self.user_quality_vector = np.full(self.n_users, 5.0, dtype=np.float32)
        for user_id, quality_pref in self.user_quality_preference.items():
            if user_id in self.user_mapping:
                user_idx = self.user_mapping[user_id]
                self.user_quality_vector[user_idx] = quality_pref

        # 6. User popularity preference vector: (n_users,)
        self.user_popularity_vector = np.ones(self.n_users, dtype=np.float32)
        for user_id, pop_pref in self.user_popularity_preference.items():
            if user_id in self.user_mapping:
                user_idx = self.user_mapping[user_id]
                self.user_popularity_vector[user_idx] = pop_pref

        print(f"  Item genre matrix: {self.item_genre_matrix.shape}")
        print(f"  Item quality vector: {self.item_quality_vector.shape}")
        print(f"  User genre matrix: {self.user_genre_matrix.shape}")
        print(f"  User quality vector: {self.user_quality_vector.shape}")

    def compute_content_score(self, user_id, movie_id):
        """Compute personalized content-based score for user-movie pair"""
        user_id = str(user_id)

        if movie_id not in self.item_features:
            return 0.0

        item_features = self.item_features[movie_id]
        score = 0.0

        # 1. Genre match score
        if user_id in self.user_genre_preferences:
            user_genres = self.user_genre_preferences[user_id]
            item_genre_vector = item_features['genre_vector']

            # Calculate genre alignment
            genre_match = 0.0
            for i, genre in enumerate(self.genre_mlb.classes_):
                if item_genre_vector[i] == 1 and genre in user_genres:
                    genre_match += user_genres[genre]

            score += genre_match * 2.0  # Genre is important

        # 2. Quality alignment score
        if user_id in self.user_quality_preference:
            user_quality_pref = self.user_quality_preference[user_id]
            item_quality = item_features['vote_average']

            if item_quality > 0:
                # Prefer items matching user's quality level
                quality_diff = abs(user_quality_pref - item_quality)
                quality_score = max(0, 1.0 - quality_diff / 5.0)
                score += quality_score * 0.5

        # 3. Popularity alignment score
        if user_id in self.user_popularity_preference:
            user_pop_pref = self.user_popularity_preference[user_id]
            item_pop = item_features['popularity']

            if item_pop > 0 and user_pop_pref > 0:
                # Log scale for popularity
                pop_diff = abs(np.log1p(user_pop_pref) - np.log1p(item_pop))
                pop_score = max(0, 1.0 - pop_diff / 5.0)
                score += pop_score * 0.3

        return score

    def compute_content_scores_vectorized(self, user_idx):
        """Compute content scores for ALL items for a given user at once (VECTORIZED)"""

        # 1. Genre match score (vectorized dot product)
        # user_genre_matrix[user_idx] is (n_genres,)
        # item_genre_matrix is (n_items, n_genres)
        # Result: (n_items,)
        genre_scores = np.dot(self.item_genre_matrix, self.user_genre_matrix[user_idx])
        genre_scores *= 2.0  # Genre is important

        # 2. Quality alignment score (vectorized)
        user_quality = self.user_quality_vector[user_idx]
        quality_diff = np.abs(user_quality - self.item_quality_vector)
        quality_scores = np.maximum(0, 1.0 - quality_diff / 5.0) * 0.5
        # Mask out items with 0 quality
        quality_scores = np.where(self.item_quality_vector > 0, quality_scores, 0)

        # 3. Popularity alignment score (vectorized)
        user_pop = self.user_popularity_vector[user_idx]
        # Only compute for non-zero popularity items and users
        pop_scores = np.zeros(self.n_items, dtype=np.float32)
        if user_pop > 0:
            valid_items = self.item_popularity_vector > 0
            item_pop_log = np.log1p(self.item_popularity_vector)
            pop_diff = np.abs(np.log1p(user_pop) - item_pop_log)
            pop_scores_tmp = np.maximum(0, 1.0 - pop_diff / 5.0) * 0.3
            pop_scores = np.where(valid_items, pop_scores_tmp, 0)

        # Combine all scores
        total_scores = genre_scores + quality_scores + pop_scores
        return total_scores

    def initialize_factors(self, n_users, n_items):
        """Initialize factors"""
        self.n_users = n_users
        self.n_items = n_items

        scale = 0.2
        self.user_factors = np.random.normal(0, scale, (n_users, self.n_factors)).astype(np.float64)
        self.item_factors = np.random.normal(0, scale, (n_items, self.n_factors)).astype(np.float64)
        self.user_bias = np.random.normal(0, 0.01, n_users).astype(np.float64)
        self.item_bias = np.random.normal(0, 0.01, n_items).astype(np.float64)

    def fit_epoch(self, user_indices, item_indices, ratings_values):
        """Train one epoch with hybrid approach"""
        n_samples = len(ratings_values)
        indices = np.random.permutation(n_samples)

        epoch_loss = 0.0

        for idx in indices:
            u = user_indices[idx]
            i = item_indices[idx]
            r = ratings_values[idx]

            # Collaborative prediction
            cf_pred = (self.global_mean +
                      self.user_bias[u] +
                      self.item_bias[i] +
                      np.dot(self.user_factors[u], self.item_factors[i]))

            # Content-based score
            user_id = self.reverse_user_mapping[u]
            movie_id = self.reverse_item_mapping[i]
            content_score = self.compute_content_score(user_id, movie_id)

            # Hybrid prediction
            pred = cf_pred + self.content_weight * content_score

            # Handle NaN predictions
            if np.isnan(pred):
                pred = self.global_mean

            pred = np.clip(pred, 0.5, 5.5)

            error = r - pred
            epoch_loss += error ** 2

            # Update CF components with gradient clipping
            bias_u_update = self.learning_rate * (error - self.regularization * self.user_bias[u])
            bias_i_update = self.learning_rate * (error - self.regularization * self.item_bias[i])

            # Clip gradients to prevent explosions
            bias_u_update = np.clip(bias_u_update, -5.0, 5.0)
            bias_i_update = np.clip(bias_i_update, -5.0, 5.0)

            self.user_bias[u] += bias_u_update
            self.item_bias[i] += bias_i_update

            user_factor_old = self.user_factors[u].copy()

            factor_u_update = self.learning_rate * (
                error * self.item_factors[i] - self.regularization * self.user_factors[u]
            )
            factor_i_update = self.learning_rate * (
                error * user_factor_old - self.regularization * self.item_factors[i]
            )

            # Clip factor updates
            factor_u_update = np.clip(factor_u_update, -5.0, 5.0)
            factor_i_update = np.clip(factor_i_update, -5.0, 5.0)

            self.user_factors[u] += factor_u_update
            self.item_factors[i] += factor_i_update

        rmse = np.sqrt(epoch_loss / n_samples)
        return rmse

    def fit(self, train_df, val_df, interactions_df, n_epochs=50, verbose=True):
        """Train hybrid model"""
        print(f"\n{'='*60}")
        print(f"Training Personalized Hybrid Recommender for {n_epochs} epochs")
        print(f"Content weight: {self.content_weight}")
        print(f"{'='*60}")

        # Build user profiles
        self.build_user_profiles(interactions_df)

        # Prepare content features
        self.prepare_content_features(interactions_df, train_df)

        # Store user interaction history
        for user_id in train_df['user_id'].unique():
            user_data = train_df[train_df['user_id'] == user_id]
            self.user_watched_items[str(user_id)] = set(user_data['movie_id'].tolist())
            self.user_ratings[str(user_id)] = dict(user_data[['movie_id', 'rating']].values)

        # Store metadata
        self.all_movie_ids = train_df['movie_id'].unique().tolist()
        self.user_ids = train_df['user_id'].unique().tolist()

        # Create mappings
        unique_users = train_df['user_id'].unique()
        unique_items = train_df['movie_id'].unique()

        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}

        self.global_mean = train_df['rating'].mean()
        self.initialize_factors(len(unique_users), len(unique_items))

        # Prepare vectorized features for fast inference
        self.prepare_vectorized_features()

        # Prepare training data - filter out NaN ratings
        train_df_clean = train_df[train_df['rating'].notna()].copy()
        user_indices = train_df_clean['user_id'].map(self.user_mapping).values
        item_indices = train_df_clean['movie_id'].map(self.item_mapping).values
        ratings_values = train_df_clean['rating'].values

        # Prepare validation data - filter out NaN ratings and unmapped users/items
        val_df_filtered = val_df[
            (val_df['user_id'].isin(self.user_mapping)) &
            (val_df['movie_id'].isin(self.item_mapping)) &
            (val_df['rating'].notna())
        ].copy()

        val_user_indices = val_df_filtered['user_id'].map(self.user_mapping).values
        val_item_indices = val_df_filtered['movie_id'].map(self.item_mapping).values
        val_ratings_values = val_df_filtered['rating'].values

        print(f"\nTraining setup:")
        print(f"  Users: {self.n_users}")
        print(f"  Items: {self.n_items}")
        print(f"  Factors: {self.n_factors}")
        print(f"  Training samples: {len(train_df_clean)}")
        print(f"  Validation samples: {len(val_df_filtered)}")
        print(f"  Global mean: {self.global_mean:.3f}")

        # Training loop
        start_time = time.time()
        self.training_history = []

        for epoch in range(n_epochs):
            epoch_start = time.time()

            train_rmse = self.fit_epoch(user_indices, item_indices, ratings_values)
            val_rmse = self.evaluate_rmse(val_user_indices, val_item_indices, val_ratings_values)

            epoch_time = time.time() - epoch_start

            self.training_history.append({
                'epoch': epoch + 1,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'epoch_time': epoch_time
            })

            if verbose and (epoch + 1) % 5 == 0:
                improvement = ""
                if val_rmse < self.best_val_rmse:
                    improvement = " *"
                    self.best_val_rmse = val_rmse
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                print(f"Epoch {epoch+1:2d}/{n_epochs}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f} ({epoch_time:.2f}s){improvement}")

                if self.patience_counter >= 10:
                    self.learning_rate *= 0.95
                    self.patience_counter = 0
                    if verbose:
                        print(f"  → Learning rate reduced to {self.learning_rate:.6f}")

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.2f} seconds!")
        print(f"Best Val RMSE: {self.best_val_rmse:.4f}")
        print(f"{'='*60}")

        return total_time

    def evaluate_rmse(self, user_indices, item_indices, ratings_values):
        """Evaluate RMSE with hybrid prediction"""
        predictions = []

        for u, i, r in zip(user_indices, item_indices, ratings_values):
            # CF prediction
            cf_pred = (self.global_mean +
                      self.user_bias[u] +
                      self.item_bias[i] +
                      np.dot(self.user_factors[u], self.item_factors[i]))

            # Content score
            user_id = self.reverse_user_mapping[u]
            movie_id = self.reverse_item_mapping[i]
            content_score = self.compute_content_score(user_id, movie_id)

            pred = cf_pred + self.content_weight * content_score

            # Handle NaN predictions by using global mean
            if np.isnan(pred):
                pred = self.global_mean

            pred = np.clip(pred, 1.0, 5.0)
            predictions.append(pred)

        rmse = np.sqrt(mean_squared_error(ratings_values, predictions))
        return rmse

    def predict_rating(self, user_id, movie_id):
        """Predict single rating with hybrid approach"""
        user_id, movie_id = str(user_id), str(movie_id)

        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            return self.global_mean

        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[movie_id]

        # CF prediction
        cf_pred = (self.global_mean +
                  self.user_bias[user_idx] +
                  self.item_bias[item_idx] +
                  np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

        # Content score
        content_score = self.compute_content_score(user_id, movie_id)

        prediction = cf_pred + self.content_weight * content_score

        # Handle NaN predictions
        if np.isnan(prediction):
            return self.global_mean

        return np.clip(prediction, 1.0, 5.0)

    def predict(self, user_id, n_recommendations=20, exclude_seen=True, diversity_boost=0.2):
        """Generate personalized hybrid recommendations (VECTORIZED - 100x faster!)"""
        user_id = str(user_id)

        if user_id not in self.user_mapping:
            # Cold start fallback
            sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in sorted_items[:n_recommendations]]

        user_idx = self.user_mapping[user_id]

        # 1. Vectorized CF scores for ALL items at once
        # Shape: (n_items,)
        cf_scores = (self.global_mean +
                     self.user_bias[user_idx] +
                     self.item_bias +
                     np.dot(self.item_factors, self.user_factors[user_idx]))

        # 2. Vectorized content scores for ALL items at once
        # Shape: (n_items,)
        content_scores = self.compute_content_scores_vectorized(user_idx)

        # 3. Combine scores
        hybrid_scores = cf_scores + self.content_weight * content_scores

        # 4. Handle NaN scores
        hybrid_scores = np.where(np.isnan(hybrid_scores), self.global_mean, hybrid_scores)

        # 5. Diversity boost (vectorized)
        popularity_penalty = diversity_boost * np.log1p(self.item_popularity_vector)
        hybrid_scores -= popularity_penalty

        # 6. Add small noise for diversity
        hybrid_scores += np.random.normal(0, 0.01, len(hybrid_scores))

        # 7. Exclude seen items
        if exclude_seen:
            watched_items = self.user_watched_items.get(user_id, set())
            for movie_id in watched_items:
                if movie_id in self.item_mapping:
                    item_idx = self.item_mapping[movie_id]
                    hybrid_scores[item_idx] = -np.inf

        # 8. Get top-K recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]

        # 9. Convert back to movie IDs
        recommendations = [self.reverse_item_mapping[idx] for idx in top_indices]

        return recommendations

    def get_model_info(self) -> dict:
        """
        Get model metadata and statistics.

        Returns:
            Dict with algorithm, n_users, n_items, n_factors, etc.
        """
        return {
            'algorithm': 'Personalized Hybrid Recommender (CF + Content)',
            'n_factors': self.n_factors,
            'regularization': self.regularization,
            'learning_rate': self.learning_rate,
            'content_weight': self.content_weight,
            'total_movies': len(self.all_movie_ids) if hasattr(self, 'all_movie_ids') else 0,
            'total_users': len(self.user_ids) if hasattr(self, 'user_ids') else 0,
            'matrix_size': f"{self.n_users}×{self.n_items}" if hasattr(self, 'n_users') else "Unknown",
            'best_val_rmse': self.best_val_rmse if hasattr(self, 'best_val_rmse') else None,
            'n_user_profiles': len(self.user_genre_preferences) if hasattr(self, 'user_genre_preferences') else 0,
            'n_items_with_features': len(self.item_features) if hasattr(self, 'item_features') else 0
        }

    def explain_recommendation(self, user_id, movie_id):
        """Explain why a movie is recommended to a user"""
        user_id = str(user_id)

        explanation = {
            'movie_id': movie_id,
            'predicted_rating': self.predict_rating(user_id, movie_id),
            'reasons': []
        }

        # Genre match
        if user_id in self.user_genre_preferences and movie_id in self.item_features:
            user_genres = self.user_genre_preferences[user_id]
            item_genres = [g for i, g in enumerate(self.genre_mlb.classes_)
                          if self.item_features[movie_id]['genre_vector'][i] == 1]
