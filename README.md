# MovieLens Recommendation System 🎬

A Netflix-style movie recommendation system built with Streamlit and Machine Learning.

## Features

- ✅ **Baseline Recommender**: Weighted Popularity (IMDB formula)
- ✅ **Content-Based ML**: Random Forest predictions on movie features
- ✅ **Discover Mode**: Get personalized recommendations with filters
- ✅ **Similar Movies**: Find movies similar to your favorites
- ✅ **TMDB Posters**: Real movie poster images
- ✅ **Netflix UI**: Dark theme with smooth animations

---

## Project Structure

```
streamlit_app/
├── app.py                      # Main Streamlit application
├── recommender.py              # Recommendation algorithms
├── utils.py                    # TMDB API & data loading
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml            # Netflix theme configuration
├── data/                       # ⚠️ CREATE THIS FOLDER
│   ├── Large_Movies.parquet
│   ├── Movie_Features.parquet
│   ├── large_links.parquet
│   └── models/
│       └── random_forest_optimized.pkl
└── README.md
```

---

## Setup Instructions

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Prepare Data Files**

Create a `data/` folder and place your files:

```bash
mkdir -p data/models
```

**Required Files:**
- `Large_Movies.parquet` → Place in `data/`
- `Movie_Features.parquet` → Place in `data/`
- `large_links.parquet` → Place in `data/` (for posters)
- `random_forest_optimized.pkl` → Place in `data/models/`

**File Locations:**
```
data/
├── Large_Movies.parquet           # Basic movie metadata
├── Movie_Features.parquet         # Engineered features for ML
├── large_links.parquet           # TMDB IDs for posters
└── models/
    └── random_forest_optimized.pkl   # Trained Random Forest model
```

### 3. **Run the App**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage Guide

### **Discover Movies Mode** 🔥

1. Select genres (optional)
2. Choose year range
3. Click "Get Recommendations"
4. See top-rated movies matching your criteria

### **Similar Movies Mode** 🎯

1. Search for a movie
2. View movie details & poster
3. Click "Find Similar Movies"
4. Get personalized similar movie suggestions

### **Algorithm Selection**

- **Baseline (Popularity)**: Uses IMDB weighted rating formula
- **Content-Based ML**: Uses Random Forest model predictions

---

## Technical Details

### **Baseline Recommender**

- **Algorithm**: IMDB Weighted Rating
- **Formula**: `WR = (v/(v+m)) × R + (m/(v+m)) × C`
  - `v` = number of votes
  - `m` = minimum votes threshold (70th percentile)
  - `R` = movie average rating
  - `C` = global mean rating
- **Similarity**: Genre overlap + year proximity

### **Content-Based ML Recommender**

- **Model**: Random Forest Regressor (optimized)
- **Features**: 62 movie features including:
  - Genre encodings (19 features)
  - Rating statistics (avg, std, median, percentiles)
  - Temporal features (age, decade, era indicators)
  - Tag genome scores (top tags, relevance)
  - Popularity metrics (rating count, frequency rank)
- **Similarity**: Cosine similarity on feature vectors
- **Performance**: Precision@10 = 0.669, RMSE = 0.756

### **Data Optimization**

- **Caching**: `@st.cache_data` for file loads
- **Lazy Loading**: Load only required columns
- **Session State**: Persist data across interactions
- **Efficient Formats**: Parquet for fast I/O

### **TMDB Integration**

- **API**: The Movie Database (TMDB)
- **Endpoint**: Movie details + poster images
- **Caching**: 1-hour TTL on API responses
- **Fallback**: Placeholder images for missing posters

---

## Performance Tips

### **For Large Datasets (33M+ ratings)**

1. **Use Parquet format** (not CSV)
2. **Sample data** for testing:
   ```python
   df = pd.read_parquet('data/Large_Movies.parquet')
   df.sample(10000).to_parquet('data/sample_movies.parquet')
   ```
3. **Reduce features** in `Movie_Features.parquet` if memory issues
4. **Disable posters** by setting `use_tmdb=False` in `utils.py`

### **Streamlit Cloud Deployment**

**Memory Limits:**
- Free tier: 1GB RAM
- **Solution**: Use smaller dataset or filter movies (e.g., only movies with 100+ ratings)

**Secrets Management:**
1. Go to Streamlit Cloud dashboard
2. Add TMDB API key in secrets:
   ```toml
   [tmdb]
   api_key = "your_api_key_here"
   ```
3. Update `utils.py`:
   ```python
   TMDB_API_KEY = st.secrets["tmdb"]["api_key"]
   ```

---

## Customization

### **Change Number of Recommendations**

In `app.py`, modify the slider range:
```python
num_recommendations = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=50,  # Increase maximum
    value=10,
    step=5
)
```

### **Add More Algorithms**

In `recommender.py`, create a new class:
```python
class CollaborativeRecommender:
    def __init__(self, ratings_df):
        # Your collaborative filtering logic
        pass
```

### **Change Theme Colors**

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#e50914"  # Netflix red
backgroundColor = "#141414"  # Dark background
```

---

## Troubleshooting

### **"Model not found" error**

**Cause**: Missing `random_forest_optimized.pkl`

**Solution**:
1. Check file exists at `data/models/random_forest_optimized.pkl`
2. Or use Baseline mode (doesn't require ML model)

### **"No movies found" error**

**Cause**: Filters too restrictive

**Solution**:
- Remove genre filters
- Expand year range
- Use different movie for similarity search

### **Slow performance**

**Cause**: Large dataset + no caching

**Solution**:
1. Restart app (clears cache)
2. Reduce dataset size
3. Use faster model (Logistic Regression instead of Random Forest)

### **Posters not loading**

**Cause**: Missing `large_links.parquet` or TMDB API issues

**Solution**:
1. Check `large_links.parquet` exists in `data/`
2. Verify TMDB API key in `utils.py`
3. Check internet connection

---

## Model Comparison Results

| Model | RMSE | Precision@10 | Use Case |
|-------|------|--------------|----------|
| **Random Forest** ✅ | 0.756 | **0.669** | Best overall |
| Logistic Regression | NaN | 0.662 | Fast inference |
| KNN | 0.808 | 0.656 | Good for similarity |

**Selected Model**: Random Forest (best Precision@10)

---

## Dataset Information

- **Source**: MovieLens Latest (33.8M ratings)
- **Movies**: 86,537
- **Users**: 330,975
- **Ratings**: 33,832,162
- **Date Range**: 1995-01-09 to 2023-07-20
- **Rating Scale**: 0.5 to 5.0 (half-star increments)

---

## API Keys

### **TMDB API** (Free)

1. Sign up at https://www.themoviedb.org/signup
2. Request API key at https://www.themoviedb.org/settings/api
3. Replace in `utils.py`:
   ```python
   TMDB_API_KEY = "your_api_key_here"
   ```

**Limits**: 1,000,000 requests/month (free tier)

---

## License

This project uses:
- **MovieLens data**: [GroupLens Research](https://grouplens.org/datasets/movielens/)
- **TMDB API**: [The Movie Database](https://www.themoviedb.org/)
- **Code**: MIT License

---

## Credits

- **Data**: GroupLens Research @ University of Minnesota
- **Posters**: The Movie Database (TMDB)
- **Framework**: Streamlit
- **ML Models**: Scikit-learn

---

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all data files are in correct locations
3. Ensure all dependencies are installed
4. Check Streamlit logs for detailed errors

---

**Built for Machine Learning Final Project** 🎓
