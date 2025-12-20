# MovieLens Recommendation System

A modern, Netflix-style movie recommendation system built with Streamlit and Machine Learning. Discover new movies or find similar ones based on your preferences using advanced recommendation algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Features

### Recommendation Modes

- **Discover Movies**: Get personalized recommendations with advanced filtering

  - Filter by movie name (partial search supported)
  - Filter by genres (multiple selection)
  - Filter by release year range
  - Two recommendation algorithms to choose from

- **Similar Movies**: Find movies similar to your favorites
  - Intelligent movie name search with autocomplete
  - View movie details and posters
  - Get personalized similar movie suggestions

### Recommendation Algorithms

- **Baseline (Popularity)**: IMDB-style weighted rating formula
  - Balances average rating with number of ratings
  - Genre and year-based similarity matching
- **Content-Based ML**: Random Forest model predictions
  - Uses 62+ engineered movie features
  - Cosine similarity for finding similar movies
  - Optimized for precision and accuracy

### User Experience

- **Netflix-inspired dark theme** with smooth animations
- **Real movie posters** from TMDB API
- **Fast search** with instant results
- **Responsive grid layout** for movie cards
- **Interactive UI** with hover effects

---




## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation & Run (3 Steps)

1. **Clone or download the repository**

   ```bash
   cd recommender_system
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will automatically open in your browser at `http://localhost:8501`

> **Note**: Make sure you have the required data files in the `data/` folder (see [Data Files Setup](#data-files-setup) below)

---

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- `streamlit`
- `pandas`
- `numpy` 
- `scikit-learn`
- `requests` 
- `pyarrow`
- `Pillow` 
- `matplotlib`
- `plotly`

### Step 2: Data Files Setup

Create the data directory structure:

```bash
mkdir -p data/models
```

**Required Data Files:**

Place the following files in the `data/` directory:

```
data/
├── large_movies.parquet      # Movie metadata (years, etc.)
├── Movie_Features.parquet    # Engineered features for model
├── large_links.parquet       # IMDB ID mappings for posters
└── models/
    └── random_forest_optimized.pkl # Pre-trained RF model
```

**File Descriptions:**

- `large_movies.parquet`: Basic movie information (title, genres, release year)
- `Movie_Features.parquet`: Extended features including ratings, tags, and engineered features
- `large_links.parquet`: Links to external databases (TMDB, IMDB) for fetching posters
- `random_forest_optimized.pkl`: Pre-trained machine learning model (optional - app works without it using baseline mode)

> **⚠️ Important**: Without these files, the application will not run. Ensure all files in the `data/` folder are present.

### Step 3: Configure TMDB API (Optional but Recommended)

For movie poster images:

1. Sign up at [TMDB](https://www.themoviedb.org/signup)
2. Get your API key from [API Settings](https://www.themoviedb.org/settings/api)
3. Update `utils.py`:
   ```python
   TMDB_API_KEY = "your_api_key_here"
   ```

> **Note**: The app works without the API key, but movie posters won't load. Free tier allows 1M requests/month.

---

## Project Structure

```
recommender_system/
├── app.py                      # Main Streamlit application
├── recommender.py              # Recommendation algorithms (Baseline & ML)
├── utils.py                    # Helper functions (data loading, posters, search)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── data/                       # Data files directory
│   ├── large_movies.parquet
│   ├── Movie_Features.parquet
│   ├── large_links.parquet
│   └── models/
│       └── random_forest_optimized.pkl
└── .gitignore                   # Git ignore rules
```

### Key Files Explained

- **`app.py`**: Main application file containing the Streamlit UI and user interactions
- **`recommender.py`**: Contains two recommendation classes:
  - `BaselineRecommender`: Popularity-based recommendations
  - `ContentBasedRecommender`: ML-based recommendations
- **`utils.py`**: Utility functions for:
  - Loading and caching data
  - Fetching movie posters from TMDB
  - Movie name search functionality
  - Formatting movie cards

---

## Usage Guide

### Discover Movies Mode
**Purpose**: Get personalized movie recommendations based on your preferences.

**Steps:**

1. Select **"Discover Movies"** mode (default)
2. (Optional) Enter a movie name to filter results (e.g., "batman", "matrix")
3. (Optional) Select genres you're interested in
4. (Optional) Adjust the release year range slider
5. Choose your preferred algorithm:
   - **Baseline (Popularity)**: Faster, based on ratings
   - **Content-Based ML**: More sophisticated, uses ML predictions
6. Select number of recommendations (5-20)
7. Click **"Get Recommendations"**

**Example Use Cases:**

- "Show me popular action movies from the 2000s"
- "Find movies with 'batman' in the title"
- "Recommend top-rated dramas from 2010-2020"

### Similar Movies Mode

**Purpose**: Find movies similar to a specific movie you like.

**Steps:**

1. Select **"Similar Movies"** mode
2. Type a movie name in the search box (e.g., "The Matrix", "Inception")
3. Browse the search results and select a movie
4. View the movie details (poster, genres, year, ratings)
5. Click **"Find Similar Movies"**
6. Browse the recommended similar movies below

**Search Tips:**

- Partial names work (e.g., "bat" finds all Batman movies)
- Case-insensitive search
- Results are sorted by relevance (exact matches first)

### Algorithm Selection

**When to use Baseline (Popularity):**

- Faster recommendations
- Good for discovering popular movies
- Works without ML model

**When to use Content-Based ML:**

- More personalized recommendations
- Better similarity matching
- Requires trained model file

---

## Technical Details

### Baseline Recommender

**Algorithm**: IMDB Weighted Rating Formula

```
WR = (v/(v+m)) × R + (m/(v+m)) × C
```

Where:

- `v` = number of votes for the movie
- `m` = minimum votes threshold (70th percentile)
- `R` = average rating of the movie
- `C` = mean rating across all movies

**Similarity Calculation:**

- Genre overlap (Jaccard similarity)
- Year proximity (inverse distance)
- Weighted score combination

### Content-Based ML Recommender

**Model**: Random Forest Regressor (Optimized)

**Features Used (62+ features):**

- Genre encodings (19 binary features)
- Rating statistics (mean, std, median, percentiles)
  - Temporal features (age, decade, era indicators)
- Tag genome scores (top tags, relevance scores)
  - Popularity metrics (rating count, frequency rank)

**Similarity Method**: Cosine similarity on feature vectors

**Performance Metrics:**

- **RMSE**: 0.756
- **Precision@10**: 0.669

### Data Processing

- **Caching**: All data files are cached using `@st.cache_data` for fast reloads
- **Session State**: Data persists across user interactions
- **Lazy Loading**: Only required columns are loaded
- **Format**: Parquet files for efficient I/O

### TMDB Integration

- **API Endpoint**: `https://api.themoviedb.org/3/movie/{tmdb_id}`
- **Caching**: 1-hour TTL on API responses
- **Fallback**: Placeholder images for missing posters
- **Rate Limiting**: Respects TMDB API limits


---

## Dataset Information

**Source**: [MovieLens Latest Dataset](https://grouplens.org/datasets/movielens/latest/)

**Statistics**:

- **Movies**: 86,537
- **Users**: 330,975
- **Ratings**: 33,832,162
- **Date Range**: January 9, 1995 - July 20, 2023
- **Rating Scale**: 0.5 to 5.0 (half-star increments)

**Data Files**:

- Movie metadata (titles, genres, years)
- User ratings and timestamps
- Tag genome scores
- Links to external databases (TMDB, IMDB)

---



### For Streamlit Cloud Deployment

**Memory Limits**:

- Free tier: 1GB RAM
- **Solution**: Use filtered/sampled dataset

**Secrets Management**:

1. Go to Streamlit Cloud dashboard
2. Add secrets:
   ```toml
   [tmdb]
   api_key = "your_api_key_here"
   ```
3. Update `utils.py`:
   ```python
   TMDB_API_KEY = st.secrets["tmdb"]["api_key"]
   ```

---

## Model Comparison

| Model                | RMSE  | Precision@10 | Speed  | Use Case                 |
| -------------------- | ----- | ------------ | ------ | ------------------------ |
| **Random Forest**  | 0.756 | **0.669**    | Medium | Best overall performance |
| Logistic Regression  | N/A   | 0.662        | Fast   | Quick recommendations    |
| KNN                  | 0.808 | 0.656        | Slow   | Good for similarity      |

**Selected Model**: Random Forest (best Precision@10 score)

---









## Project Information

**Built for**: Machine Learning Final Project

**Technologies**: Python, Streamlit, Scikit-learn, Pandas, NumPy

**Dataset**: MovieLens Latest (33.8M ratings)

---


