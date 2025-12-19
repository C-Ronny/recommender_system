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
        rated_movies = self.movies_df[self.movies_df['num_ratings'].notna()].copy()
        
        C = rated_movies['avg_rating'].mean()
        m = rated_movies['num_ratings'].quantile(0.70)
        
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
    
    def recommend(self, n=10, genres=None, year_range=None, movie_name=None):
        """Get top-N recommendations based on weighted popularity"""
        filtered_df = self.movies_df[self.movies_df['weighted_score'].notna()].copy()
        
        # Filter by movie name if provided
        if movie_name and len(movie_name.strip()) > 0:
            search_query = movie_name.strip().lower()
            title_mask = filtered_df['title'].astype(str).str.lower().str.contains(search_query, na=False, regex=False)
            filtered_df = filtered_df[title_mask]
        
        if genres:
            genre_mask = filtered_df['genres'].apply(
                lambda x: any(g in x for g in genres) if isinstance(x, str) else False
            )
            filtered_df = filtered_df[genre_mask]
        
        if year_range:
            filtered_df = filtered_df[
                (filtered_df['release_year'] >= year_range[0]) &
                (filtered_df['release_year'] <= year_range[1])
            ]
        
        if len(filtered_df) == 0:
            return None
        
        recommendations = filtered_df.nlargest(n, 'weighted_score')
        return recommendations[['movieId', 'title', 'genres', 'release_year', 
                               'avg_rating', 'num_ratings', 'weighted_score']]
    
    def find_similar(self, movie_id, n=10):
        """Find similar movies based on genre overlap"""
        ref_movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if ref_movie.empty:
            return None
        
        ref_genres = set(ref_movie.iloc[0]['genres'].split('|'))
        ref_year = ref_movie.iloc[0]['release_year']
        
        def genre_similarity(genres_str):
            movie_genres = set(genres_str.split('|'))
            return len(ref_genres & movie_genres) / len(ref_genres | movie_genres)
        
        candidates = self.movies_df[
            (self.movies_df['movieId'] != movie_id) &
            (self.movies_df['weighted_score'].notna())
        ].copy()
        
        candidates['genre_sim'] = candidates['genres'].apply(genre_similarity)
        
        if pd.notna(ref_year):
            candidates['year_proximity'] = 1 / (1 + abs(candidates['release_year'] - ref_year) / 10)
        else:
            candidates['year_proximity'] = 0.5
        
        candidates['similarity_score'] = (
            0.7 * candidates['genre_sim'] + 
            0.3 * candidates['year_proximity']
        )
        
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
        self.expected_features = None
        self._load_model()
        self._prepare_features()
    
    def _load_model(self):
        """Load pre-trained Random Forest model"""
        model_path = Path("data/models/random_forest_optimized.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            if hasattr(self.model, 'n_features_in_'):
                self.expected_features = self.model.n_features_in_
            
            print(f"✓ Loaded ML model successfully")
        except FileNotFoundError:
            print(f"⚠ Model not found, using baseline mode")
            self.model = None
            self.expected_features = None
    
    def _prepare_features(self):
        """Identify feature columns for ML model"""
        exclude_cols = ['movieId', 'title', 'genres', 'genre_list_clean', 
                       'imdbId', 'tmdbId', 'weighted_score', 'top_genome_tag']
        
        potential_features = [col for col in self.movie_features.columns 
                             if col not in exclude_cols]
        
        # Convert timestamp columns to numeric
        for col in potential_features:
            if pd.api.types.is_datetime64_any_dtype(self.movie_features[col]):
                self.movie_features[f'{col}_numeric'] = (
                    self.movie_features[col].astype('int64') / 10**9 / 86400
                ).fillna(0)
                exclude_cols.append(col)
        
        # Only keep numeric columns
        self.feature_cols = []
        for col in potential_features:
            if col not in exclude_cols:
                if pd.api.types.is_numeric_dtype(self.movie_features[col]):
                    self.feature_cols.append(col)
        
        # Add converted timestamp columns
        self.feature_cols.extend([col for col in self.movie_features.columns 
                                 if col.endswith('_numeric')])
        
        self.feature_cols = sorted(self.feature_cols)
        
        # Handle missing/infinite values
        self.movie_features[self.feature_cols] = (
            self.movie_features[self.feature_cols]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        
        print(f"✓ Prepared {len(self.feature_cols)} features for recommendations")
    
    def _get_feature_matrix(self, movie_ids=None):
        """Extract feature matrix for given movie IDs"""
        if movie_ids is None:
            df_subset = self.movie_features
        else:
            df_subset = self.movie_features[self.movie_features['movieId'].isin(movie_ids)]
        
        features = df_subset[self.feature_cols].values
        
        # Pad or truncate to match expected features
        if self.expected_features and features.shape[1] != self.expected_features:
            if features.shape[1] < self.expected_features:
                padding = np.zeros((features.shape[0], self.expected_features - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :self.expected_features]
        
        return features
    
    def recommend_popular(self, n=10, genres=None, year_range=None, movie_name=None):
        """Get top-N recommendations using ML predictions"""
        if self.model is None:
            baseline = BaselineRecommender(self.movies_df)
            return baseline.recommend(n, genres, year_range, movie_name)
        
        candidates = self.movie_features[
            self.movie_features['num_ratings'].notna()
        ].copy()
        
        # Filter by movie name if provided
        if movie_name and len(movie_name.strip()) > 0:
            search_query = movie_name.strip().lower()
            title_mask = candidates['title'].astype(str).str.lower().str.contains(search_query, na=False, regex=False)
            candidates = candidates[title_mask]
        
        if genres:
            genre_mask = candidates['genres'].apply(
                lambda x: any(g in x for g in genres) if isinstance(x, str) else False
            )
            candidates = candidates[genre_mask]
        
        if year_range:
            candidates = candidates[
                (candidates['release_year'] >= year_range[0]) &
                (candidates['release_year'] <= year_range[1])
            ]
        
        if len(candidates) == 0:
            return None
        
        try:
            X = self._get_feature_matrix(candidates['movieId'].values)
            predictions = self.model.predict(X)
            candidates['predicted_rating'] = predictions
            
            candidates['ml_score'] = (
                0.7 * candidates['predicted_rating'] / 5.0 +
                0.3 * np.log1p(candidates['num_ratings']) / np.log1p(candidates['num_ratings'].max())
            )
            
            recommendations = candidates.nlargest(n, 'ml_score')
            return recommendations[['movieId', 'title', 'genres', 'release_year', 
                                  'avg_rating', 'num_ratings', 'predicted_rating']]
        
        except Exception as e:
            print(f"ML prediction error: {e}, using baseline")
            baseline = BaselineRecommender(self.movies_df)
            return baseline.recommend(n, genres, year_range)
    
    def find_similar(self, movie_id, n=10):
        """Find similar movies using cosine similarity on features"""
        ref_movie = self.movie_features[
            self.movie_features['movieId'] == movie_id
        ]
        
        if ref_movie.empty:
            return None
        
        ref_features = self._get_feature_matrix([movie_id])
        
        candidates = self.movie_features[
            (self.movie_features['movieId'] != movie_id) &
            (self.movie_features['num_ratings'].notna())
        ].copy()
        
        if len(candidates) == 0:
            return None
        
        candidate_features = self._get_feature_matrix(candidates['movieId'].values)
        
        try:
            similarities = cosine_similarity(ref_features, candidate_features)[0]
            candidates['similarity'] = similarities
            
            candidates['final_score'] = (
                0.8 * candidates['similarity'] +
                0.2 * (candidates['avg_rating'] / 5.0)
            )
            
            similar_movies = candidates.nlargest(n, 'final_score')
            return similar_movies[['movieId', 'title', 'genres', 'release_year', 
                                  'avg_rating', 'num_ratings', 'similarity']]
        
        except Exception as e:
            print(f"Similarity error: {e}")
            return None
