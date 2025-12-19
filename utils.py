import pandas as pd
import streamlit as st
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")  # Fallback to public key if not set
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_poster(movie_id, links_df, use_tmdb=True):
    """
    Fetch movie poster from TMDB
    
    Args:
        movie_id: MovieLens movie ID
        links_df: DataFrame with movieId, imdbId, tmdbId
        use_tmdb: If False, return placeholder
    
    Returns:
        Poster URL or placeholder
    """
    if not use_tmdb or links_df is None:
        return "https://via.placeholder.com/300x450?text=No+Poster"
    
    # Get TMDB ID
    link_data = links_df[links_df['movieId'] == movie_id]
    
    if link_data.empty or pd.isna(link_data.iloc[0].get('tmdbId')):
        return "https://via.placeholder.com/300x450?text=No+Poster"
    
    tmdb_id = int(link_data.iloc[0]['tmdbId'])
    
    # Fetch from TMDB API
    try:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {"api_key": TMDB_API_KEY}
        
        response = requests.get(url, params=params, timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"
        
        return "https://via.placeholder.com/300x450?text=No+Poster"
    
    except Exception:
        return "https://via.placeholder.com/300x450?text=No+Poster"


def format_movie_card(movie, poster_url):
    """
    Generate HTML for Netflix-style movie card
    
    Args:
        movie: Movie row from DataFrame
        poster_url: URL to movie poster
    
    Returns:
        HTML string for movie card
    """
    title = movie['title']
    genres = movie['genres'].replace('|', ', ') if '|' in str(movie['genres']) else movie['genres']
    
    # Extract year from title if available
    year = ""
    if pd.notna(movie.get('release_year')):
        year = f"({int(movie['release_year'])})"
    
    # Rating display
    rating_html = ""
    if 'avg_rating' in movie and pd.notna(movie['avg_rating']):
        rating = movie['avg_rating']
        rating_html = f'<div class="movie-rating">⭐ {rating:.1f}/5.0</div>'
    elif 'predicted_rating' in movie and pd.notna(movie['predicted_rating']):
        rating = movie['predicted_rating']
        rating_html = f'<div class="movie-rating">🤖 {rating:.1f}/5.0</div>'
    
    # Number of ratings
    num_ratings_html = ""
    if 'num_ratings' in movie and pd.notna(movie['num_ratings']):
        num_ratings = int(movie['num_ratings'])
        num_ratings_html = f'<div style="font-size: 11px; color: #666;">{num_ratings:,} ratings</div>'
    
    card_html = f"""
    <div class="movie-card">
        <img src="{poster_url}" 
             style="width: 100%; border-radius: 4px; aspect-ratio: 2/3; object-fit: cover;"
             onerror="this.src='https://via.placeholder.com/300x450?text=No+Poster'">
        <div class="movie-title">{title} {year}</div>
        <div class="movie-genres">{genres}</div>
        {rating_html}
        {num_ratings_html}
    </div>
    """
    
    return card_html


@st.cache_data(show_spinner=False)
def load_data_cached():
    """
    Load all required data files with caching
    
    Returns:
        movies_df, movie_features, links_df
    """
    data_path = Path("data")
    
    # Load movies
    movies_df = pd.read_parquet(data_path / "Large_Movies.parquet")
    
    # Load movie features
    movie_features = pd.read_parquet(data_path / "Movie_Features.parquet")
    
    # Extract release year from title if not in features
    if 'release_year' not in movies_df.columns:
        movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)$').astype(float)
    
    # Merge features with movies
    movies_df = movies_df.merge(
        movie_features[['movieId', 'avg_rating', 'num_ratings', 'release_year']],
        on='movieId',
        how='left',
        suffixes=('', '_feat')
    )
    
    # Use feature release_year if available
    if 'release_year_feat' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_year_feat'].fillna(movies_df['release_year'])
        movies_df.drop('release_year_feat', axis=1, inplace=True)
    
    # Load links (for TMDB posters)
    try:
        links_df = pd.read_parquet(data_path / "large_links.parquet")
    except FileNotFoundError:
        try:
            links_df = pd.read_csv(data_path / "links.csv")
        except FileNotFoundError:
            st.warning("Links file not found. Movie posters will be unavailable.")
            links_df = None
    
    return movies_df, movie_features, links_df


def format_metrics_row(metrics_dict):
    """Format metrics in a nice row display"""
    cols = st.columns(len(metrics_dict))
    
    for col, (label, value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(label, value)
