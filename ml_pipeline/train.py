"""
Model Training Module

Handles model training orchestration and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

from .model import ImprovedSVDRecommendationModel, PersonalizedHybridRecommender
from .feature_engineering import (
    build_user_item_matrix,
    calculate_popular_movies,
    calculate_global_statistics
)

def train_test_split_temporal(interactions_df: pd.DataFrame,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train and test sets.

    For recommendation systems, should ideally use temporal splitting
    (train on older interactions, test on newer ones) to avoid data leakage.

    Args:
        interactions_df: DataFrame with user_id, movie_id, rating
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    # Check if temporal column exists
    if 'ts' in interactions_df.columns or 'timestamp' in interactions_df.columns:
        # Temporal split
        ts_col = 'ts' if 'ts' in interactions_df.columns else 'timestamp'
        sorted_df = interactions_df.sort_values(ts_col)
        split_idx = int(len(sorted_df) * (1 - test_size))
        train_df = sorted_df.iloc[:split_idx].copy()
        test_df = sorted_df.iloc[split_idx:].copy()
    else:
        # Random split (fallback)
        train_df, test_df = train_test_split(
            interactions_df,
            test_size=test_size,
            random_state=random_state
        )

    return train_df, test_df


def train_svd_model(train_df: pd.DataFrame,
                config: dict,
                log_to_wandb: bool = False) -> ImprovedSVDRecommendationModel:
    """
    Train SVD collaborative filtering model.

    Steps:
    1. Build user-item rating matrix
    2. Calculate global statistics (mean, biases)
    3. Perform SVD decomposition
    4. Store learned factors and biases in model

    Args:
        train_df: Training interactions DataFrame
        config: SVD configuration (n_factors, regularization, etc.)
        log_to_wandb: Whether to log metrics to Weights & Biases

    Returns:
        Trained ImprovedSVDRecommendationModel

    Example:
        >>> config = {"n_factors": 100, "regularization": 0.01}
        >>> model = train_svd_model(train_df, config)
        >>> print(model.n_users, model.n_items)
        21061 7227

    Note:
        Orchestrates features.py and model.py modules
    """
    # Extract config
    n_factors = config.get('n_factors', 100)
    regularization = config.get('regularization', 0.01)
    n_popular = config.get('n_popular_movies', 50)

    # Build user-item matrix and mappings
    print("Building user-item matrix...")
    matrix, mappings = build_user_item_matrix(train_df)
    print(f"Matrix shape: {matrix.shape}")

    # Calculate popular movies for cold start
    print("Calculating popular movies...")
    popular_movies = calculate_popular_movies(train_df, n_top=n_popular)

    # Calculate global statistics (biases)
    print("Calculating global statistics...")
    global_stats = calculate_global_statistics(train_df, mappings)
    print(f"Global mean rating: {global_stats['global_mean']:.3f}")

    # Subtract biases from matrix for SVD
    # Adjust ratings: r_adjusted = r - global_mean - user_bias - item_bias
    user_indices = train_df['user_id'].map(mappings['user_mapping']).values
    item_indices = train_df['movie_id'].map(mappings['item_mapping']).values

    adjusted_ratings = (
        train_df['rating'].values -
        global_stats['global_mean'] -
        np.array([global_stats['user_biases'][i] for i in user_indices]) -
        np.array([global_stats['item_biases'][i] for i in item_indices])
    )

    # Rebuild matrix with adjusted ratings
    from scipy.sparse import csr_matrix
    adjusted_matrix = csr_matrix(
        (adjusted_ratings, (user_indices, item_indices)),
        shape=(mappings['n_users'], mappings['n_items']),
        dtype=np.float32
    )

    # Initialize and train model
    print(f"Training SVD with {n_factors} factors...")
    model = ImprovedSVDRecommendationModel(
        n_factors=n_factors,
        regularization=regularization
    )

    model.fit(
        ratings_matrix=adjusted_matrix,
        mappings=mappings,
        global_stats=global_stats,
        popular_movies=popular_movies
    )

    print("Training completed!")

    # Log to W&B if requested
    if log_to_wandb:
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "n_users": model.n_users,
                    "n_items": model.n_items,
                    "n_ratings": len(train_df),
                    "sparsity": 1 - (len(train_df) / (model.n_users * model.n_items)),
                    "global_mean": model.global_mean,
                    "n_factors": n_factors
                })
        except ImportError:
            print("Warning: wandb not installed, skipping logging")

    return model


def train_hybrid_model(train_df: pd.DataFrame,
                       interactions_df: pd.DataFrame,
                       config: dict,
                       val_df: pd.DataFrame = None,
                       log_to_wandb: bool = False) -> PersonalizedHybridRecommender:
    """
    Train Personalized Hybrid Recommender model.

    Combines collaborative filtering with content-based features (genres,
    popularity, ratings) and personalized user preferences.

    Steps:
    1. Split train_df into train/val if val_df not provided
    2. Initialize PersonalizedHybridRecommender with config
    3. Train model with SGD and early stopping
    4. Return trained model

    Args:
        train_df: Training interactions DataFrame (user_id, movie_id, rating/minutes)
        interactions_df: Full interactions with metadata (genres, vote_average, etc.)
        config: Hybrid configuration (n_factors, learning_rate, etc.)
        val_df: Optional validation DataFrame (if None, split from train_df)
        log_to_wandb: Whether to log metrics to Weights & Biases

    Returns:
        Trained PersonalizedHybridRecommender

    Example:
        >>> config = {"n_factors": 100, "learning_rate": 0.015}
        >>> model = train_hybrid_model(train_df, interactions_df, config)
        >>> recs = model.predict("user_123", n_recommendations=10)

    Note:
        Requires interactions_df to contain metadata columns:
        - genres: comma-separated genre strings
        - vote_average: movie rating (0-10)
        - popularity: movie popularity score
        - runtime: movie duration in minutes (for watch-time conversion)
    """
    # Extract config
    n_factors = config.get('n_factors', 100)
    learning_rate = config.get('learning_rate', 0.015)
    regularization = config.get('regularization', 0.005)
    content_weight = config.get('content_weight', 0.25)
    n_epochs = config.get('n_epochs', 50)
    min_interactions = config.get('min_interactions', 3)

    print("Initializing Personalized Hybrid Recommender...")
    print(f"  n_factors: {n_factors}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  regularization: {regularization}")
    print(f"  content_weight: {content_weight}")
    print(f"  n_epochs: {n_epochs}")

    # Create validation split if not provided
    if val_df is None:
        print("Creating validation split (80/20)...")
        train_subset, val_df = train_test_split_temporal(
            train_df,
            test_size=0.2,
            random_state=42
        )
    else:
        train_subset = train_df

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_df)}")

    # Initialize model
    model = PersonalizedHybridRecommender(
        n_factors=n_factors,
        learning_rate=learning_rate,
        regularization=regularization,
        content_weight=content_weight
    )

    # Train model
    print(f"Training Hybrid Model for {n_epochs} epochs...")
    model.fit(
        train_df=train_subset,
        val_df=val_df,
        interactions_df=interactions_df,
        n_epochs=n_epochs
    )

    print("Training completed!")

    # Log to W&B if requested
    if log_to_wandb:
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "n_users": model.n_users,
                    "n_items": model.n_items,
                    "n_ratings": len(train_subset),
                    "n_factors": n_factors,
                    "learning_rate": learning_rate,
                    "content_weight": content_weight,
                })
        except ImportError:
            print("Warning: wandb not installed, skipping logging")

    return model


def train_neural_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Train Neural Hybrid Recommender (Phase 3).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .model_neural import NeuralHybridRecommender
    from .feature_engineering_phase3 import prepare_neural_features

    print("🔧 Preparing features for neural model...")
    full_df, encoders = prepare_neural_features(train_df)
    n_users = len(encoders["user_encoder"].classes_)
    n_items = len(encoders["movie_encoder"].classes_)
    n_meta = full_df.shape[1] - 3  # exclude user_idx, movie_idx, rating

    # Prepare tensors
    users = torch.tensor(full_df["user_idx"].values, dtype=torch.long)
    items = torch.tensor(full_df["movie_idx"].values, dtype=torch.long)
    ratings = torch.tensor(full_df["rating"].values, dtype=torch.float32)
    metadata = torch.tensor(full_df.drop(columns=["user_idx","movie_idx","rating"]).values, dtype=torch.float32)

    dataset = TensorDataset(users, items, metadata, ratings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = NeuralHybridRecommender(n_users, n_items, n_meta, embed_dim=config.get("embed_dim", 64))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print(f"🚀 Training Neural Hybrid Model on {len(dataset)} samples...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for u, i, m, r in loader:
            u, i, m, r = u.to(device), i.to(device), m.to(device), r.to(device)
            optimizer.zero_grad()
            preds = model(u, i, m)
            loss = loss_fn(preds, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")

    print("✅ Training complete.")
    return model, encoders
