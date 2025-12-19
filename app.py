import streamlit as st
import pandas as pd
import numpy as np
from recommender import BaselineRecommender, ContentBasedRecommender
from utils import get_movie_poster, format_movie_card, load_data_cached
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    .movie-card {
        background: #2a2a2a;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(229,9,20,0.4);
    }
    .movie-title {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        margin: 8px 0 4px 0;
    }
    .movie-genres {
        font-size: 12px;
        color: #999;
    }
    .movie-rating {
        font-size: 14px;
        color: #e50914;
        font-weight: 600;
    }
    .section-header {
        color: #e50914;
        font-size: 24px;
        font-weight: 700;
        margin: 30px 0 20px 0;
        border-left: 4px solid #e50914;
        padding-left: 12px;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #ffffff !important;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #f40612;
    }
    div[data-testid="stMetricValue"] {
        color: #e50914;
    }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.movies_df = None
    st.session_state.movie_features = None
    st.session_state.links_df = None
    st.session_state.baseline_rec = None
    st.session_state.content_rec = None

@st.cache_data(show_spinner=False)
def initialize_app():
    """Load all data and initialize recommenders"""
    try:
        movies_df, movie_features, links_df = load_data_cached()
        
        baseline_rec = BaselineRecommender(movies_df)
        content_rec = ContentBasedRecommender(movies_df, movie_features, links_df)
        
        return movies_df, movie_features, links_df, baseline_rec, content_rec
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

if not st.session_state.initialized:
    with st.spinner("🎬 Loading MovieLens data..."):
        (st.session_state.movies_df, 
         st.session_state.movie_features,
         st.session_state.links_df,
         st.session_state.baseline_rec,
         st.session_state.content_rec) = initialize_app()
        
        if st.session_state.movies_df is not None:
            st.session_state.initialized = True

if st.session_state.initialized:
    movies_df = st.session_state.movies_df
    baseline_rec = st.session_state.baseline_rec
    content_rec = st.session_state.content_rec
    
    st.markdown('<h1 style="color: #e50914; text-align: center; margin-bottom: 10px;">🎬 MovieLens Recommender</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #999; margin-bottom: 40px;">Powered by Machine Learning</p>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
        
        recommendation_mode = st.radio(
            "Recommendation Mode",
            ["🔥 Discover Movies", "🎯 Similar Movies"],
            help="Choose how you want to find movies"
        )
        
        st.markdown("---")
        
        recommender_type = st.radio(
            "Algorithm",
            ["🌟 Baseline (Popularity)", "🤖 Content-Based ML"],
            help="Baseline uses weighted ratings, ML uses Random Forest model"
        )
        
        st.markdown("---")
        
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Movies", f"{len(movies_df):,}")
        st.metric("Total Ratings", "33.8M")
        st.metric("Users", "331K")
    
    if "🔥 Discover" in recommendation_mode:
        st.markdown('<div class="section-header">🔥 Discover New Movies</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            all_genres = sorted(set([g for genres in movies_df['genres'].str.split('|') 
                                     for g in genres if g != '(no genres listed)']))
            
            selected_genres = st.multiselect(
                "Select Genres (optional)",
                options=all_genres,
                default=None,
                help="Leave empty for all genres"
            )
        
        with col2:
            min_year = int(movies_df['release_year'].min())
            max_year = int(movies_df['release_year'].max())
            
            year_range = st.slider(
                "Release Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(1990, max_year),
                help="Filter movies by release year"
            )
        
        if st.button("🎬 Get Recommendations", key="discover_btn"):
            with st.spinner("Finding perfect movies for you..."):
                if "Baseline" in recommender_type:
                    recommendations = baseline_rec.recommend(
                        n=num_recommendations,
                        genres=selected_genres if selected_genres else None,
                        year_range=year_range
                    )
                else:
                    recommendations = content_rec.recommend_popular(
                        n=num_recommendations,
                        genres=selected_genres if selected_genres else None,
                        year_range=year_range
                    )
                
                if recommendations is not None and len(recommendations) > 0:
                    st.markdown(f'<div class="section-header">✨ Top {len(recommendations)} Recommendations</div>', 
                              unsafe_allow_html=True)
                    
                    cols = st.columns(5)
                    for idx, (_, movie) in enumerate(recommendations.iterrows()):
                        with cols[idx % 5]:
                            poster_url = get_movie_poster(movie['movieId'], st.session_state.links_df)
                            card_html = format_movie_card(movie, poster_url)
                            st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.warning("No movies found matching your criteria. Try different filters.")
    
    else:
        st.markdown('<div class="section-header">🎯 Find Similar Movies</div>', unsafe_allow_html=True)
        
        movie_titles = movies_df['title'].tolist()
        
        selected_movie = st.selectbox(
            "Search for a movie",
            options=movie_titles,
            index=None,
            placeholder="Type to search...",
            help="Search by movie title"
        )
        
        if selected_movie:
            selected_movie_data = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                poster_url = get_movie_poster(selected_movie_data['movieId'], st.session_state.links_df)
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
            
            with col2:
                st.markdown(f"### {selected_movie_data['title']}")
                st.markdown(f"**Genres:** {selected_movie_data['genres'].replace('|', ', ')}")
                st.markdown(f"**Year:** {int(selected_movie_data['release_year']) if pd.notna(selected_movie_data['release_year']) else 'Unknown'}")
                
                if 'avg_rating' in selected_movie_data:
                    st.metric("Average Rating", f"{selected_movie_data['avg_rating']:.2f} ⭐")
                if 'num_ratings' in selected_movie_data:
                    st.metric("Total Ratings", f"{int(selected_movie_data['num_ratings']):,}")
            
            with col3:
                if st.button("🔍 Find Similar Movies", key="similar_btn"):
                    with st.spinner("Analyzing movie features..."):
                        if "Baseline" in recommender_type:
                            similar_movies = baseline_rec.find_similar(
                                movie_id=selected_movie_data['movieId'],
                                n=num_recommendations
                            )
                        else:
                            similar_movies = content_rec.find_similar(
                                movie_id=selected_movie_data['movieId'],
                                n=num_recommendations
                            )
                        
                        if similar_movies is not None and len(similar_movies) > 0:
                            st.markdown(f'<div class="section-header">🎬 Movies Similar to "{selected_movie}"</div>', 
                                      unsafe_allow_html=True)
                            
                            cols = st.columns(5)
                            for idx, (_, movie) in enumerate(similar_movies.iterrows()):
                                with cols[idx % 5]:
                                    poster_url = get_movie_poster(movie['movieId'], st.session_state.links_df)
                                    card_html = format_movie_card(movie, poster_url)
                                    st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            st.warning("Could not find similar movies. Try a different movie.")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with Streamlit | Data from MovieLens | Posters from TMDB</p>
        <p>Machine Learning Final Project</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to initialize the application. Please check your data files.")
