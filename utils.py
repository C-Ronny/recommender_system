import pandas as pd
import streamlit as st
import requests
from pathlib import Path

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_poster(movie_id, links_df, use_tmdb=True):
    """Fetch movie poster from TMDB"""
    if not use_tmdb or links_df is None:
        return "https://via.placeholder.com/300x450?text=No+Poster"
    
    link_data = links_df[links_df['movieId'] == movie_id]
    
    if link_data.empty or pd.isna(link_data.iloc[0].get('tmdbId')):
        return "https://via.placeholder.com/300x450?text=No+Poster"
    
    tmdb_id = int(link_data.iloc[0]['tmdbId'])
    
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
    """Generate HTML for Netflix-style movie card"""
    title = movie['title']
    genres = movie['genres'].replace('|', ', ') if '|' in str(movie['genres']) else movie['genres']
    
    year = ""
    if pd.notna(movie.get('release_year')):
        year = f"({int(movie['release_year'])})"
    
    rating_html = ""
    if 'avg_rating' in movie and pd.notna(movie['avg_rating']):
        rating = movie['avg_rating']
        rating_html = f'<div class="movie-rating">⭐ {rating:.1f}/5.0</div>'
    elif 'predicted_rating' in movie and pd.notna(movie['predicted_rating']):
        rating = movie['predicted_rating']
        rating_html = f'<div class="movie-rating">🤖 {rating:.1f}/5.0</div>'
    
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
    """Load all required data files with caching"""
    data_path = Path("data")
    
    movies_df = pd.read_parquet(data_path / "Large_Movies.parquet")
    
    movie_features = pd.read_parquet(data_path / "Movie_Features.parquet")
    
    if 'release_year' not in movies_df.columns:
        movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)$').astype(float)
    
    movies_df = movies_df.merge(
        movie_features[['movieId', 'avg_rating', 'num_ratings', 'release_year']],
        on='movieId',
        how='left',
        suffixes=('', '_feat')
    )
    
    if 'release_year_feat' in movies_df.columns:
        movies_df['release_year'] = movies_df['release_year_feat'].fillna(movies_df['release_year'])
        movies_df.drop('release_year_feat', axis=1, inplace=True)
    
    try:
        links_df = pd.read_parquet(data_path / "large_links.parquet")
    except FileNotFoundError:
        try:
            links_df = pd.read_csv(data_path / "links.csv")
        except FileNotFoundError:
            links_df = None
    
    return movies_df, movie_features, links_df


def format_metrics_row(metrics_dict):
    """Format metrics in a nice row display"""
    cols = st.columns(len(metrics_dict))
    
    for col, (label, value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(label, value)


def search_movies_by_name(movies_df, search_query, max_results=20):
    """
    Search for movies by name with case-insensitive partial matching.
    Supports both substring matching and word-based matching for better results.
    
    Args:
        movies_df: DataFrame containing movie data with 'title' column
        search_query: String to search for in movie titles
        max_results: Maximum number of results to return
    
    Returns:
        DataFrame with matching movies, sorted by relevance
    """
    if not search_query or len(search_query.strip()) == 0:
        return pd.DataFrame()
    
    search_query = search_query.strip().lower()
    search_words = search_query.split()
    
    # Ensure 'title' column exists
    if 'title' not in movies_df.columns:
        return pd.DataFrame()
    
    # Filter movies where title contains the search query (case-insensitive)
    # Use contains with regex=False for better performance
    title_lower = movies_df['title'].astype(str).str.lower()
    mask = title_lower.str.contains(search_query, na=False, regex=False)
    matches = movies_df[mask].copy()
    
    if len(matches) == 0:
        return pd.DataFrame()
    
    # Calculate relevance scores using the filtered matches
    matches_title_lower = matches['title'].astype(str).str.lower()
    matches['is_exact_match'] = (matches_title_lower == search_query).astype(int)
    matches['starts_with'] = matches_title_lower.str.startswith(search_query).astype(int)
    
    # Word-based matching: count how many search words appear in the title
    def count_matching_words(title):
        title_lower_str = str(title).lower()
        return sum(1 for word in search_words if word in title_lower_str)
    
    matches['word_matches'] = matches['title'].apply(count_matching_words)
    matches['word_match_ratio'] = matches['word_matches'] / len(search_words)
    
    # Calculate final relevance score
    matches['relevance_score'] = (
        matches['is_exact_match'] * 100 +
        matches['starts_with'] * 50 +
        matches['word_match_ratio'] * 30 +
        (matches['title'].str.len() < 50).astype(int) * 5  # Prefer shorter titles (usually more relevant)
    )
    
    # Sort by relevance score, then by title
    matches = matches.sort_values(
        by=['relevance_score', 'title'],
        ascending=[False, True]
    )
    
    # Drop temporary columns
    matches = matches.drop(columns=['is_exact_match', 'starts_with', 'word_matches', 
                                   'word_match_ratio', 'relevance_score'], errors='ignore')
    
    return matches.head(max_results)
