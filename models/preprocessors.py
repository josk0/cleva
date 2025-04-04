"""Custom classes used in sklean pipeline"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeFeatureSplitter(BaseEstimator, TransformerMixin):
    """ Split datetime columns into numerical columns with name suffix _year, _month, _day respectively """
    def __init__(self):
        self.datetime_cols = None
    
    def fit(self, X, y=None):
        # Identify datetime columns during fit
        self.datetime_cols = X.select_dtypes(include=['datetime64']).columns
        return self
    
    def transform(self, X):
        # Apply your datetime transformations
        X_transformed = X.copy()
        for col in self.datetime_cols:
            X_transformed[f'{col}_year'] = X[col].dt.year
            X_transformed[f'{col}_month'] = X[col].dt.month
            X_transformed[f'{col}_day'] = X[col].dt.day
        
        # Drop original datetime columns
        X_transformed = X_transformed.drop(columns=self.datetime_cols)
        return X_transformed
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class DatetimeFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, cyclical=True):
        """
        Extract features from datetime columns.
        
        Parameters:
        -----------
        features : list, optional
            List of features to extract. Options: 'year', 'month', 'day', 'weekday', 
            'quarter', 'hour', 'minute', 'second', 'is_weekend', 'is_month_start', 
            'is_month_end', 'day_of_year'
            If None, will extract all time-related features.
        cyclical : bool, default=True
            Whether to encode cyclical features (month, day, weekday, hour, etc.) using 
            sine/cosine transformation to preserve their cyclical nature.
        """
        # Set default features to encode if none given
        self.features = features if features else [
            'year', 'month', 'day', 'weekday'
        ]
        self.cyclical = cyclical
        self._cyclical_features = ['month', 'day', 'weekday', 'hour', 'minute', 'second', 'day_of_year']
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        self.datetime_cols = X.select_dtypes(include=['datetime64']).columns
        return self
    
    def _encode_cyclical(self, feature_values, max_value):
        """Encode cyclical features using sine and cosine transformations."""
        # Normalize to [0, 2*pi]
        values_scaled = 2 * np.pi * feature_values / max_value
        # Return sine and cosine
        return np.sin(values_scaled), np.cos(values_scaled)
    
    def transform(self, X):
        """Transform datetime columns into numerical features."""
        # Ensure X is a pandas Series or a copy to avoid modifying the original
        X_copy = X.copy() if isinstance(X, pd.Series) else X.squeeze()
        # Handle both 1D and 2D arrays
        # if hasattr(X, 'iloc') and hasattr(X, 'copy'):  # It's already a pandas object
        #     X_copy = X.copy()
        #     print("already pandas!")
        # else:
        #     # Check if it's a 2D array with a single column and convert appropriately
        #     if isinstance(X, np.ndarray) and X.ndim > 1:
        #         print("2d array!")
        #         if X.shape[1] == 1:
        #             X_copy = pd.Series(X.flatten())
        #         else:
        #             raise ValueError(f"Expected array with 1 column, got shape {X.shape}")
        #     else:
        #         print("1d array!")
        #         X_copy = pd.Series(X)
        
        # Convert string to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(X_copy):
            X_copy = pd.to_datetime(X_copy, errors='coerce')

        # Initialize an empty DataFrame for storing the extracted features
        X_transformed = pd.DataFrame(index=X_copy.index)
        
        # Define max values for cyclical features
        max_values = {
            'month': 12,
            'day': 31,
            'weekday': 7,
            'hour': 24,
            'minute': 60,
            'second': 60,
            'day_of_year': 366
        }
        
        # Extract and transform features
        all_features = []
        
        if 'year' in self.features:
            X_transformed['year'] = X_copy.dt.year
            all_features.append('year')
            
        for feat in self.features:
            if feat == 'year':
                continue  # Already handled above
                
            if feat == 'month' and hasattr(X_copy.dt, 'month'):
                feature_values = X_copy.dt.month
            elif feat == 'day' and hasattr(X_copy.dt, 'day'):
                feature_values = X_copy.dt.day
            elif feat == 'weekday' and hasattr(X_copy.dt, 'weekday'):
                feature_values = X_copy.dt.weekday
            elif feat == 'quarter' and hasattr(X_copy.dt, 'quarter'):
                feature_values = X_copy.dt.quarter
            elif feat == 'hour' and hasattr(X_copy.dt, 'hour'):
                feature_values = X_copy.dt.hour
            elif feat == 'minute' and hasattr(X_copy.dt, 'minute'):
                feature_values = X_copy.dt.minute
            elif feat == 'second' and hasattr(X_copy.dt, 'second'):
                feature_values = X_copy.dt.second
            elif feat == 'is_weekend' and hasattr(X_copy.dt, 'weekday'):
                feature_values = (X_copy.dt.weekday >= 5).astype(int)
            elif feat == 'is_month_start' and hasattr(X_copy.dt, 'is_month_start'):
                feature_values = X_copy.dt.is_month_start.astype(int)
            elif feat == 'is_month_end' and hasattr(X_copy.dt, 'is_month_end'):
                feature_values = X_copy.dt.is_month_end.astype(int)
            elif feat == 'day_of_year' and hasattr(X_copy.dt, 'dayofyear'):
                feature_values = X_copy.dt.dayofyear
            else:
                continue  # Skip if feature is not available
            
            # Apply cyclical encoding for relevant features
            if self.cyclical and feat in self._cyclical_features:
                sin_vals, cos_vals = self._encode_cyclical(feature_values, max_values.get(feat, 1))
                X_transformed[f'{feat}_sin'] = sin_vals
                X_transformed[f'{feat}_cos'] = cos_vals
                all_features.extend([f'{feat}_sin', f'{feat}_cos'])
            else:
                X_transformed[feat] = feature_values
                all_features.append(feat)
        
        # Store feature names for get_feature_names_out
        self.feature_names_out_ = all_features
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        return self.feature_names_out_
