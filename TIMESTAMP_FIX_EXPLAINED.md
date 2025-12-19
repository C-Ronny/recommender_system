# Timestamp Error Fix - Detailed Explanation

## The Problem

**Error Message:**
```
ML prediction error: float() argument must be a string or a real number, not 'Timestamp'
```

**Root Cause:**
Your `Movie_Features.parquet` file contains timestamp/datetime columns (e.g., `first_rating_date`, `last_rating_date`) that cannot be directly passed to machine learning models. The Random Forest model expects numeric values only.

## The Solution

The fix automatically:
1. **Detects** all timestamp/datetime columns
2. **Converts** them to numeric values (days since Unix epoch)
3. **Excludes** the original timestamp columns
4. **Validates** all features are numeric before prediction

### Code Changes in `recommender.py`

#### Before (Broken):
```python
def _prepare_features(self):
    exclude_cols = ['movieId', 'title', 'genres', ...]
    
    self.feature_cols = [col for col in self.movie_features.columns 
                        if col not in exclude_cols]
    
    self.movie_features[self.feature_cols] = self.movie_features[self.feature_cols].fillna(0)
```

#### After (Fixed):
```python
def _prepare_features(self):
    exclude_cols = ['movieId', 'title', 'genres', ...]
    
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
    
    # Handle missing/infinite values
    self.movie_features[self.feature_cols] = (
        self.movie_features[self.feature_cols]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )
```

## How to Apply the Fix

### Option 1: Download Updated Package (Recommended)
1. Download `streamlit_app_fixed.zip`
2. Extract and replace your existing files
3. Run: `streamlit run app.py`

### Option 2: Use Auto-Fix Script
```bash
# In your streamlit_app folder
python fix_timestamp_error.py
```

### Option 3: Manual Update
Replace the `_prepare_features` method in `recommender.py` with the fixed version above.

## Verification

After applying the fix, you should see:
```
✓ Loaded model from data/models/random_forest_optimized.pkl
✓ Prepared 62 numeric features for ML model
```

The Content-Based ML recommendations will now work correctly!

## Technical Details

### Timestamp Conversion Formula
```python
days_since_epoch = timestamp.astype('int64') / 10**9 / 86400
```

- `astype('int64')`: Converts to nanoseconds since Unix epoch
- `/ 10**9`: Converts nanoseconds to seconds
- `/ 86400`: Converts seconds to days

### Additional Safeguards
1. **Type checking**: Validates all features are numeric
2. **NaN handling**: Replaces with 0
3. **Inf handling**: Replaces infinite values with 0
4. **Logging**: Prints number of features prepared

## Testing

Test both recommendation modes:

1. **Discover Mode**:
   - Select genres: Action, Adventure
   - Year range: 1990-2024
   - Click "Get Recommendations"
   - Should return top movies without error

2. **Similar Movies Mode**:
   - Search: "Toy Story (1995)"
   - Click "Find Similar Movies"
   - Should return similar movies without error

If you still see errors, check:
- Model file location: `data/models/random_forest_optimized.pkl`
- Feature file: `Movie_Features.parquet` has required columns
- Run: `python verify_setup.py` to validate setup

## Common Issues After Fix

### "Model not found" warning
- Place your trained model at: `data/models/random_forest_optimized.pkl`
- App will fall back to Baseline mode automatically

### "No movies found" error
- Try different filters (remove genres, expand year range)
- Not a code error - just no movies match criteria

### Slow performance
- First load caches data (~30 seconds)
- Subsequent requests are fast
- Restart app to clear cache if needed

## Need Help?

Run the verification script:
```bash
python verify_setup.py
```

This checks:
- ✅ Data files present
- ✅ Correct file structure
- ✅ Required columns exist
- ✅ Model file location
