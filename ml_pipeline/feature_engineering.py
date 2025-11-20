"""
Feature Engineering Module

Handles creation of features for model training:
- User-item rating matrix construction
- User/item index mappings
- Sparse matrix representation
- Popularity features

"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict

'''
This is for SVD model
'''
def build_user_item_matrix(interactions_df: pd.DataFrame) -> Tuple[csr_matrix, Dict]:
    """
    Build sparse user-item rating matrix and create index mappings.

    Args:
        interactions_df: DataFrame with columns: user_id, movie_id, rating

    Returns:
        Tuple of (sparse_matrix, mappings_dict) where:
        - sparse_matrix: scipy.sparse.csr_matrix of shape (n_users, n_items)
        - mappings_dict: {
            'user_mapping': {user_id: index},
            'item_mapping': {movie_id: index},
            'reverse_user_mapping': {index: user_id},
            'reverse_item_mapping': {index: movie_id},
            'n_users': int,
            'n_items': int
        }

    Example:
        >>> matrix, mappings = build_user_item_matrix(interactions_df)
        >>> print(matrix.shape)
        (21061, 7227)
        >>> print(mappings['n_users'])
        21061
    """
    # Create mappings
    mappings = create_user_item_mappings(interactions_df)

    # Map user and movie IDs to integer indices
    user_indices = interactions_df['user_id'].map(mappings['user_mapping']).values
    item_indices = interactions_df['movie_id'].map(mappings['item_mapping']).values
    ratings = interactions_df['rating'].values

    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (ratings, (user_indices, item_indices)),
        shape=(mappings['n_users'], mappings['n_items']),
        dtype=np.float32
    )

    return sparse_matrix, mappings


def create_user_item_mappings(interactions_df: pd.DataFrame) -> Dict:
    """
    Create bidirectional mappings between user/item IDs and integer indices.

    Args:
        interactions_df: DataFrame with user_id and movie_id columns

    Returns:
        Dict with user_mapping, item_mapping, reverse_user_mapping, reverse_item_mapping
    """
    # Get unique users and items
    unique_users = interactions_df['user_id'].unique()
    unique_items = interactions_df['movie_id'].unique()

    # Create forward mappings (ID -> index)
    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx for idx, item in enumerate(unique_items)}

    # Create reverse mappings (index -> ID)
    reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
    reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}

    return {
        'user_mapping': user_mapping,
        'item_mapping': item_mapping,
        'reverse_user_mapping': reverse_user_mapping,
        'reverse_item_mapping': reverse_item_mapping,
        'n_users': len(unique_users),
        'n_items': len(unique_items)
    }


def calculate_popular_movies(interactions_df: pd.DataFrame,
                             n_top: int = 50) -> list:
    """
    Calculate most popular movies based on rating count and average rating.

    Popularity score = rating_count * avg_rating

    Args:
        interactions_df: DataFrame with movie_id and rating columns
        n_top: Number of top popular movies to return

    Returns:
        List of movie_ids sorted by popularity (descending)

    Note:
        Extracted from model/model2/model2.2.py lines 232-243
    """
    movie_popularity = interactions_df.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    }).round(2)

    movie_popularity.columns = ['rating_count', 'avg_rating']

    # Calculate popularity score
    movie_popularity['popularity_score'] = (
        movie_popularity['rating_count'] * movie_popularity['avg_rating']
    )

    # Get top N popular movies
    popular_movies = (movie_popularity
                     .sort_values('popularity_score', ascending=False)
                     .head(n_top)
                     .index.tolist())

    return popular_movies


def calculate_global_statistics(ratings_df: pd.DataFrame, mappings: Dict) -> Dict:
    """
    Calculate global statistics for bias modeling.

    Args:
        ratings_df: DataFrame with user_id, movie_id, rating
        mappings: Dict with user_mapping and item_mapping

    Returns:
        Dict with:
        - global_mean: float (mean rating across all interactions)
        - user_biases: np.array (user bias = user_mean - global_mean)
        - item_biases: np.array (item bias = item_mean - global_mean)

    Note:
        Extracted from model/model2/model2.2.py lines 260-276
    """
    # Calculate global mean
    global_mean = ratings_df['rating'].mean()

    # Calculate per-user and per-item means
    user_means = ratings_df.groupby('user_id')['rating'].mean()
    item_means = ratings_df.groupby('movie_id')['rating'].mean()

    # Initialize bias arrays
    n_users = len(mappings['user_mapping'])
    n_items = len(mappings['item_mapping'])

    user_biases = np.zeros(n_users)
    item_biases = np.zeros(n_items)

    # Calculate user biases
    for user_id, mean_rating in user_means.items():
        if user_id in mappings['user_mapping']:
            user_idx = mappings['user_mapping'][user_id]
            user_biases[user_idx] = mean_rating - global_mean

    # Calculate item biases
    for item_id, mean_rating in item_means.items():
        if item_id in mappings['item_mapping']:
            item_idx = mappings['item_mapping'][item_id]
            item_biases[item_idx] = mean_rating - global_mean

    return {
        'global_mean': global_mean,
        'user_biases': user_biases,
        'item_biases': item_biases
    }


def filter_sparse_interactions(interactions_df: pd.DataFrame,
                               min_user_interactions: int = 5,
                               min_item_interactions: int = 3) -> pd.DataFrame:
    """
    Filter out users and items with too few interactions.

    Helps reduce sparsity and improve model quality.

    Args:
        interactions_df: DataFrame with user_id, movie_id, rating
        min_user_interactions: Minimum number of interactions per user
        min_item_interactions: Minimum number of interactions per movie

    Returns:
        Filtered DataFrame

    Note:
        Extracted from model/model2/model2.2.py lines 189-206
    """
    df = interactions_df.copy()

    # Filter users
    if min_user_interactions > 0:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]

    # Filter movies
    if min_item_interactions > 0:
        movie_counts = df['movie_id'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_item_interactions].index
        df = df[df['movie_id'].isin(valid_movies)]

    return df


'''
This is for NN model
'''
def prepare_neural_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()

    # 1️⃣ Label Encode IDs
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df["user_idx"] = user_encoder.fit_transform(df["user_id"])
    df["movie_idx"] = movie_encoder.fit_transform(df["movie_id"])

    # 2️⃣ Clean numeric columns
    numeric_cols = ["runtime", "vote_average", "vote_count", "popularity", "revenue", "budget"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # 3️⃣ Multi-hot encode genres
    mlb = MultiLabelBinarizer()
    df["genres_list"] = df["genres"].apply(
        lambda g: [x.strip() for x in g.split(",")] if isinstance(g, str) else []
    )
    genre_features = pd.DataFrame(
        mlb.fit_transform(df["genres_list"]),
        columns=[f"genre_{g}" for g in mlb.classes_]
    )

    # 4️⃣ Scale numeric
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_vals, columns=[f"scaled_{c}" for c in numeric_cols])

    # 5️⃣ Combine
    full_df = pd.concat(
        [df[["user_idx", "movie_idx", "rating"]],
         genre_features.reset_index(drop=True),
         scaled_df.reset_index(drop=True)], axis=1
    )

    encoders = {
        "user_encoder": user_encoder,
        "movie_encoder": movie_encoder,
        "genre_binarizer": mlb,
        "scaler": scaler
    }

    return full_df, encoders

