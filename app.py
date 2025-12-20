import streamlit as st
import pandas as pd
import numpy as np
from recommender import BaselineRecommender, ContentBasedRecommender
from utils import get_movie_poster, format_movie_card, load_data_cached, search_movies_by_name
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
    .similar-movies-section {
        margin-top: 40px;
        padding-top: 20px;
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
    st.session_state.selected_movie_id = None
    st.session_state.similar_movies = None

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
    
    
    with st.sidebar:
        st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)
        
        recommendation_mode = st.radio(
            "Recommendation Mode",
            ["Discover Movies", "Similar Movies"],
            help="Choose how you want to find movies"
        )
        
        st.markdown("---")
        
        recommender_type = st.radio(
            "Algorithm",
            ["Baseline (Popularity)", "Content-Based ML"],
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
        st.markdown("### Dataset Info")
        st.metric("Total Movies", f"{len(movies_df):,}")
        st.metric("Total Ratings", "33.8M")
        st.metric("Users", "331K")
    
    if "Discover Movies" in recommendation_mode:
        st.markdown('<div class="section-header">Discover New Movies</div>', unsafe_allow_html=True)
        
        # Optional movie name search filter
        movie_search_filter = st.text_input(
            "🔍 Filter by movie name (optional)",
            placeholder="Type to filter movies by name...",
            help="Optionally filter recommendations by movie name",
            key="discover_movie_search"
        )
        
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
        
        if st.button("Get Recommendations", key="discover_btn"):
            with st.spinner("Finding perfect movies for you..."):
                # Apply movie name filter if provided (pass to recommender)
                movie_name_filter = movie_search_filter.strip() if movie_search_filter and len(movie_search_filter.strip()) > 0 else None
                
                if "Baseline" in recommender_type:
                    recommendations = baseline_rec.recommend(
                        n=num_recommendations,
                        genres=selected_genres if selected_genres else None,
                        year_range=year_range,
                        movie_name=movie_name_filter
                    )
                else:
                    recommendations = content_rec.recommend_popular(
                        n=num_recommendations,
                        genres=selected_genres if selected_genres else None,
                        year_range=year_range,
                        movie_name=movie_name_filter
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
                    error_msg = "No movies found matching your criteria."
                    if movie_name_filter:
                        error_msg += f" No movies found with '{movie_name_filter}' in the title"
                        if selected_genres:
                            error_msg += f" in the selected genres"
                        if year_range:
                            error_msg += f" in the year range {year_range[0]}-{year_range[1]}"
                        error_msg += ". Try:"
                        error_msg += "\n- Using a different movie name or partial name"
                        error_msg += "\n- Removing genre filters"
                        error_msg += "\n- Expanding the year range"
                    else:
                        error_msg += " Try adjusting your filters."
                    st.warning(error_msg)
    
    else:
        st.markdown('<div class="section-header">Find Similar Movies</div>', unsafe_allow_html=True)
        
        # Movie search input
        search_query = st.text_input(
            "Search for a movie 🔍",
            placeholder="Type movie name to search (e.g., 'batman', 'matrix', 'titanic')...",
            help="Start typing to search for movies by name",
            key="movie_search_input"
        )
        
        selected_movie = None
        selected_movie_data = None
        
        if search_query and len(search_query.strip()) > 0:
            # Search for movies matching the query
            with st.spinner(f"Searching for movies matching '{search_query}'..."):
                search_results = search_movies_by_name(movies_df, search_query, max_results=100)
            
            if len(search_results) > 0:
                st.success(f"Found {len(search_results)} movie(s) matching '{search_query}'")
                
                # Show a preview of search results
                preview_limit = min(10, len(search_results))
                st.markdown(f"**Preview of top {preview_limit} results:**")
                
                # Display results in a compact list format
                preview_results = search_results.head(preview_limit)
                result_list_html = "<div style='max-height: 200px; overflow-y: auto; background: #1a1a1a; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                for idx, (_, movie) in enumerate(preview_results.iterrows()):
                    year = f" ({int(movie['release_year'])})" if pd.notna(movie.get('release_year')) else ""
                    rating = f" ⭐ {movie['avg_rating']:.1f}" if pd.notna(movie.get('avg_rating')) else ""
                    result_list_html += f"<div style='padding: 5px; border-bottom: 1px solid #333;'>{idx+1}. <strong>{movie['title']}</strong>{year}{rating}</div>"
                result_list_html += "</div>"
                st.markdown(result_list_html, unsafe_allow_html=True)
                
                # Display search results in a more visible way
                st.markdown("**Select a movie from the results below:**")
                
                # Create a more user-friendly selection interface
                result_titles = search_results['title'].tolist()
                
                # Show first 30 results in a selectbox, with option to see more
                display_limit = min(30, len(result_titles))
                display_titles = result_titles[:display_limit]
                
                if len(result_titles) > display_limit:
                    st.info(f"Showing top {display_limit} of {len(result_titles)} results. Refine your search to see more specific matches.")
                
                selected_title = st.selectbox(
                    f"Choose a movie ({len(search_results)} found)",
                    options=display_titles,
                    index=None,
                    help="Select a movie from the search results",
                    key="movie_select_from_search"
                )
                
                if selected_title:
                    selected_movie = selected_title
                    selected_movie_data = movies_df[movies_df['title'] == selected_title].iloc[0]
            else:
                st.warning(f"No movies found matching '{search_query}'. Try a different search term.")
                st.info("💡 Tip: Try searching with partial movie names (e.g., 'bat' for Batman movies)")
        elif search_query and len(search_query.strip()) == 0:
            st.info("👆 Start typing a movie name to search...")
        
        if selected_movie and selected_movie_data is not None:
            selected_movie_data = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            # Store selected movie in session state
            if 'selected_movie_id' not in st.session_state or st.session_state.selected_movie_id != selected_movie_data['movieId']:
                st.session_state.selected_movie_id = selected_movie_data['movieId']
                st.session_state.similar_movies = None  # Clear previous similar movies
            
            col1, col2 = st.columns([1, 2])
            
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
                
                col_rating, col_count = st.columns(2)
                with col_rating:
                    if 'avg_rating' in selected_movie_data:
                        st.metric("Average Rating", f"{selected_movie_data['avg_rating']:.2f} ⭐")
                with col_count:
                    if 'num_ratings' in selected_movie_data:
                        st.metric("Total Ratings", f"{int(selected_movie_data['num_ratings']):,}")
                
                if st.button("🔍 Find Similar Movies", key="similar_btn", use_container_width=True):
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
                            st.session_state.similar_movies = similar_movies
                            st.rerun()
                        else:
                            st.warning("Could not find similar movies. Try a different movie.")
                            st.session_state.similar_movies = None
            
            # Display similar movies in a full-width section below
            if st.session_state.similar_movies is not None and len(st.session_state.similar_movies) > 0:
                st.markdown("---")
                st.markdown(f'<div class="section-header similar-movies-section">🎬 Movies Similar to "{selected_movie}"</div>', 
                          unsafe_allow_html=True)
                
                # Display movies in a responsive grid (5 columns on large screens)
                num_movies = len(st.session_state.similar_movies)
                cols = st.columns(5)
                
                for idx, (_, movie) in enumerate(st.session_state.similar_movies.iterrows()):
                    with cols[idx % 5]:
                        poster_url = get_movie_poster(movie['movieId'], st.session_state.links_df)
                        card_html = format_movie_card(movie, poster_url)
                        st.markdown(card_html, unsafe_allow_html=True)
                
                # Add some spacing at the bottom
                st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with Streamlit | Data from MovieLens | Posters from TMDB</p>
        
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to initialize the application. Please check your data files.")
