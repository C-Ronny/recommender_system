#!/usr/bin/env python3
"""
Quick fix for Timestamp error in Content-Based ML model
Run this script to update your recommender.py file
"""

import sys
from pathlib import Path

def apply_fix():
    """Apply the timestamp handling fix to recommender.py"""
    
    recommender_file = Path("recommender.py")
    
    if not recommender_file.exists():
        print("❌ recommender.py not found in current directory")
        print("   Make sure you're in the streamlit_app folder")
        return False
    
    print("🔧 Applying timestamp fix to recommender.py...")
    
    # Read current file
    with open(recommender_file, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'is_datetime64_any_dtype' in content:
        print("✅ File already has the fix applied!")
        return True
    
    # Find and replace the _prepare_features method
    old_method = """    def _prepare_features(self):
        \"\"\"Identify feature columns for ML model\"\"\"
        exclude_cols = ['movieId', 'title', 'genres', 'genre_list_clean', 
                       'imdbId', 'tmdbId', 'weighted_score']
        
        self.feature_cols = [col for col in self.movie_features.columns 
                            if col not in exclude_cols]
        
        # Handle missing values
        self.movie_features[self.feature_cols] = self.movie_features[self.feature_cols].fillna(0)"""
    
    new_method = """    def _prepare_features(self):
        \"\"\"Identify feature columns for ML model\"\"\"
        exclude_cols = ['movieId', 'title', 'genres', 'genre_list_clean', 
                       'imdbId', 'tmdbId', 'weighted_score']
        
        # Get all potential feature columns
        potential_features = [col for col in self.movie_features.columns 
                             if col not in exclude_cols]
        
        # Convert timestamp columns to numeric (days since epoch)
        for col in potential_features:
            if pd.api.types.is_datetime64_any_dtype(self.movie_features[col]):
                # Convert to days since Unix epoch
                self.movie_features[f'{col}_numeric'] = (
                    self.movie_features[col].astype('int64') / 10**9 / 86400
                ).fillna(0)
                exclude_cols.append(col)  # Exclude original timestamp
        
        # Only keep numeric columns
        self.feature_cols = []
        for col in potential_features:
            if col not in exclude_cols:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(self.movie_features[col]):
                    self.feature_cols.append(col)
        
        # Add converted timestamp columns
        self.feature_cols.extend([col for col in self.movie_features.columns 
                                 if col.endswith('_numeric')])
        
        # Handle missing values
        self.movie_features[self.feature_cols] = self.movie_features[self.feature_cols].fillna(0)
        
        # Handle infinite values
        self.movie_features[self.feature_cols] = self.movie_features[self.feature_cols].replace(
            [np.inf, -np.inf], 0
        )
        
        print(f"✓ Prepared {len(self.feature_cols)} numeric features for ML model")"""
    
    if old_method not in content:
        print("⚠️  Could not find the method to replace.")
        print("   Your file may have been modified. Please download the updated version.")
        return False
    
    # Apply fix
    content = content.replace(old_method, new_method)
    
    # Backup original
    backup_file = recommender_file.with_suffix('.py.backup')
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Write fixed version
    with open(recommender_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fix applied successfully!")
    print(f"   Backup saved to: {backup_file}")
    print("\n🎬 You can now run: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("MovieLens Recommender - Timestamp Error Fix")
    print("=" * 60)
    print("\nThis fixes the error:")
    print('  "float() argument must be a string or a real number, not \'Timestamp\'"\n')
    
    success = apply_fix()
    
    if not success:
        print("\n❌ Fix could not be applied automatically.")
        print("   Download the updated streamlit_app.zip package.")
    
    print("=" * 60)
