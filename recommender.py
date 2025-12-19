import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class BaselineRecommender:
    """Weighted Popularity Baseline (IMDB formula)"""
    
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self._prepare_weighted_scores()
    
    def _prepare_weighted_scores(self):
        """Calculate IMDB weighted rating for all movies"""
        # Filter movies with ratings data
        rated_movies = self.movies_df[self.movies_df['num_ratings'].notna()].copy()
        
        # IMDB weighted rating formula
        C = rated_movies['avg_rating'].mean()  # Mean rating across all movies
        m = rated_movies['num_ratings'].quantile(0.70)  # Minimum votes (70th percentile)
        
        def weighted_rating(row):
            v = row['num_ratings']
            R = row['avg_rating']
            return (v / (v + m) * R) + (m / (v + m) * C)
        
        rated_movies['weighted_score'] = rated_movies.apply(weighted_rating, axis=1)
        self.movies_df = self.movies_df.merge(
            rated_movies[['movieId', 'weighted_score']], 
            on='movieId', 
            how='left'
        )
    
    def recommend(self, n=10, genres=None, year_range=None):
        """Get top-N recommendations based on weighted popularity"""
        filtered_df = self.movies_df[self.movies_df['weighted_score'].notna()].copy()
        
        # Apply genre filter
        if genres:
            genre_mask = filtered_df['genres'].apply(
                lambda x: any(g in x for g in genres)
            )
            filtered_df = filtered_df[genre_mask]
        
        # Apply year filter
        if year_range:
            filtered_df = filtered_df[
                (filtered_df['release_year'] >= year_range[0]) &
                (filtered_df['release_year'] <= year_range[1])
            ]
        
        # Sort by weighted score
        recommendations = filtered_df.nlargest(n, 'weighted_score')
        return recommendations[['movieId', 'title', 'genres', 'release_year', 
                               'avg_rating', 'num_ratings', 'weighted_score']]
    
    def find_similar(self, movie_id, n=10):
        """Find similar movies based on genre overlap"""
        # Get reference movie
        ref_movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if ref_movie.empty:
            return None
        
        ref_genres = set(ref_movie.iloc[0]['genres'].split('|'))
        ref_year = ref_movie.iloc[0]['release_year']
        
        # Calculate similarity scores
        def genre_similarity(genres_str):
            movie_genres = set(genres_str.split('|'))
            return len(ref_genres & movie_genres) / len(ref_genres | movie_genres)
        
        candidates = self.movies_df[
            (self.movies_df['movieId'] != movie_id) &
            (self.movies_df['weighted_score'].notna())
        ].copy()
        
        candidates['genre_sim'] = candidates['genres'].apply(genre_similarity)
        
        # Add year proximity bonus
        if pd.notna(ref_year):
            candidates['year_proximity'] = 1 / (1 + abs(candidates['release_year'] - ref_year) / 10)
        else:
            candidates['year_proximity'] = 0.5
        
        # Combined score
        candidates['similarity_score'] = (
            0.7 * candidates['genre_sim'] + 
            0.3 * candidates['year_proximity']
        )
        
        # Sort by similarity and weighted score
        candidates['final_score'] = (
            0.6 * candidates['similarity_score'] + 
            0.4 * candidates['weighted_score'] / candidates['weighted_score'].max()
        )
        
        similar_movies = candidates.nlargest(n, 'final_score')
        return similar_movies[['movieId', 'title', 'genres', 'release_year', 
                              'avg_rating', 'num_ratings']]


class ContentBasedRecommender:
    """ML-based Content Recommender using Random Forest"""
    
    def __init__(self, movies_df, movie_features, links_df):
        self.movies_df = movies_df
        self.movie_features = movie_features
        self.links_df = links_df
        self.model = None
        self.feature_cols = None
        self._load_model()
        self._prepare_features()
    
    def _load_model(self):
        """Load pre-trained Random Forest model"""
        model_path = Path("data/models/random_forest_optimized.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"⚠ Model not found at {model_path}")
            print("  Place your trained model at data/models/random_forest_optimized.pkl")
            self.model = None
    
    def _prepare_features(self):
        """Identify feature columns for ML model"""
        exclude_cols = ['movieId', 'title', 'genres', 'genre_list_clean', 
                       'imdbId', 'tmdbId', 'weighted_score']
        
        self.feature_cols = [col for col in self.movie_features.columns 
                            if col not in exclude_cols]
        
        # Handle missing values
        self.movie_features[self.feature_cols] = self.movie_features[self.feature_cols].fillna(0)
    
    def _get_feature_matrix(self, movie_ids=None):
        """Extract feature matrix for given movie IDs"""
        if movie_ids is None:
            features = self.movie_features[self.feature_cols].values
        else:
            features = self.movie_features[
                self.movie_features['movieId'].isin(movie_ids)
            ][self.feature_cols].values
        
        return features
    
    def recommend_popular(self, n=10, genres=None, year_range=None):
        """Get top-N recommendations using ML predictions"""
        if self.model is None:
            # Fallback to baseline
            st.warning("ML model not loaded. Using baseline recommendations.")
            baseline = BaselineRecommender(self.movies_df)
            return baseline.recommend(n, genres, year_range)
        
        # Filter candidates
        candidates = self.movie_features[
            self.movie_features['num_ratings'].notna()
        ].copy()
        
        # Apply filters
        if genres:
            genre_mask = candidates['genres'].apply(
                lambda x: any(g in x for g in genres)
            )
            candidates = candidates[genre_mask]
        
        if year_range:
            candidates = candidates[
                (candidates['release_year'] >= year_range[0]) &
                (candidates['release_year'] <= year_range[1])
            ]
        
        if len(candidates) == 0:
            return None
        
        # Predict ratings for all candidates
        X = candidates[self.feature_cols].fillna(0).values
        
        try:
            predictions = self.model.predict(X)
            candidates['predicted_rating'] = predictions
            
            # Combine with popularity
            candidates['ml_score'] = (
                0.7 * candidates['predicted_rating'] / 5.0 +
                0.3 * np.log1p(candidates['num_ratings']) / np.log1p(candidates['num_ratings'].max())
            )
            
            recommendations = candidates.nlargest(n, 'ml_score')
            return recommendations[['movieId', 'title', 'genres', 'release_year', 
                                  'avg_rating', 'num_ratings', 'predicted_rating']]
        
        except Exception as e:
            st.error(f"ML prediction error: {e}")
            return None
    
    def find_similar(self, movie_id, n=10):
        """Find similar movies using cosine similarity on features"""
        # Get reference movie features
        ref_movie = self.movie_features[
            self.movie_features['movieId'] == movie_id
        ]
        
        if ref_movie.empty:
            return None
        
        ref_features = ref_movie[self.feature_cols].fillna(0).values
        
        # Get all other movies
        candidates = self.movie_features[
            (self.movie_features['movieId'] != movie_id) &
            (self.movie_features['num_ratings'].notna())
        ].copy()
        
        if len(candidates) == 0:
            return None
        
        # Calculate cosine similarity
        candidate_features = candidates[self.feature_cols].fillna(0).values
        similarities = cosine_similarity(ref_features, candidate_features)[0]
        
        candidates['similarity'] = similarities
        
        # Combine similarity with quality (avg_rating)
        candidates['final_score'] = (
            0.8 * candidates['similarity'] +
            0.2 * (candidates['avg_rating'] / 5.0)
        )
        
        similar_movies = candidates.nlargest(n, 'final_score')
        return similar_movies[['movieId', 'title', 'genres', 'release_year', 
                              'avg_rating', 'num_ratings', 'similarity']]
