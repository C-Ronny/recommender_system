#!/usr/bin/env python3
"""
Verify that all required data files are in place for the Streamlit app
"""
import sys
from pathlib import Path
import pandas as pd

def check_file(filepath, required=True):
    """Check if file exists and display info"""
    if filepath.exists():
        try:
            if filepath.suffix == '.parquet':
                df = pd.read_parquet(filepath)
                print(f"✅ {filepath.name}")
                print(f"   Shape: {df.shape}, Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
                return True
            elif filepath.suffix == '.pkl':
                print(f"✅ {filepath.name}")
                print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
                return True
        except Exception as e:
            print(f"⚠️  {filepath.name} - Error reading: {e}")
            return False
    else:
        if required:
            print(f"❌ {filepath.name} - MISSING (Required)")
        else:
            print(f"⚠️  {filepath.name} - Missing (Optional)")
        return False

def main():
    print("=" * 60)
    print("MovieLens Recommender - Data Setup Verification")
    print("=" * 60)
    
    data_dir = Path("data")
    models_dir = data_dir / "models"
    
    # Check directories
    print("\n📁 Directory Structure:")
    if not data_dir.exists():
        print(f"❌ data/ directory does not exist!")
        print("   → Create it with: mkdir data")
        return False
    else:
        print(f"✅ data/ directory exists")
    
    if not models_dir.exists():
        print(f"⚠️  data/models/ directory does not exist")
        print("   → Create it with: mkdir data/models")
    else:
        print(f"✅ data/models/ directory exists")
    
    # Check required files
    print("\n📊 Required Data Files:")
    required_files = [
        data_dir / "large_movies.parquet",
        data_dir / "Movie_Features.parquet",
    ]
    
    optional_files = [
        data_dir / "large_links.parquet",
        models_dir / "random_forest_optimized.pkl",
    ]
    
    all_good = True
    for filepath in required_files:
        if not check_file(filepath, required=True):
            all_good = False
    
    print("\n📊 Optional Files:")
    for filepath in optional_files:
        check_file(filepath, required=False)
    
    # Data validation
    print("\n🔍 Data Validation:")
    
    try:
        movies = pd.read_parquet(data_dir / "large_movies.parquet")
        features = pd.read_parquet(data_dir / "Movie_Features.parquet")
        
        # Check columns
        required_movie_cols = ['movieId', 'title', 'genres']
        required_feature_cols = ['movieId', 'avg_rating', 'num_ratings']
        
        missing_cols = [col for col in required_movie_cols if col not in movies.columns]
        if missing_cols:
            print(f"⚠️  large_movies.parquet missing columns: {missing_cols}")
        else:
            print(f"✅ large_movies.parquet has required columns")
        
        missing_cols = [col for col in required_feature_cols if col not in features.columns]
        if missing_cols:
            print(f"⚠️  Movie_Features.parquet missing columns: {missing_cols}")
        else:
            print(f"✅ Movie_Features.parquet has required columns")
        
        # Check data quality
        common_ids = set(movies['movieId']) & set(features['movieId'])
        print(f"✅ {len(common_ids):,} movies with features")
        
        if len(common_ids) < 1000:
            print(f"⚠️  Very few movies have features ({len(common_ids)})")
        
    except Exception as e:
        print(f"❌ Error validating data: {e}")
        all_good = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ Setup complete! You can run the app with:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some issues found. Please fix them before running the app.")
        print("\nQuick Setup Guide:")
        print("1. Create data directory: mkdir -p data/models")
        print("2. Copy files to data/:")
        print("   - large_movies.parquet")
        print("   - Movie_Features.parquet")
        print("   - large_links.parquet (for posters)")
        print("3. Copy model to data/models/:")
        print("   - random_forest_optimized.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()
