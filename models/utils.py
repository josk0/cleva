from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from models.preprocessors import DatetimeFeatureSplitter, DatetimeFeatureEncoder


def get_pipeline(run):
  if run is None:
    raise ValueError("Run must be provided.")
  
  # Create preprocessor based on model
  if "RandomForest" in run.model_name:
    print("RandomForest model detected.")
    preprocessor = ColumnTransformer(
        transformers=[
            ('datetime', DatetimeFeatureEncoder(), make_column_selector(dtype_include='datetime64[ns]')),
            ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include='object'))
        ])
  elif "TabPFN" in run.model_name:
    print("TabPFN model detected.")
    preprocessor = ColumnTransformer(
        transformers=[
            ('datetime', DatetimeFeatureSplitter(), make_column_selector(dtype_include='datetime64[ns]')),
        ])
  else:
    # Default preprocessing for unrecognized models
    print(f"Using default preprocessing for {run.model_name}")
    preprocessor = ColumnTransformer(
        transformers=[
            ('datetime', DatetimeFeatureSplitter(), make_column_selector(dtype_include='datetime64[ns]')),
            ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include='object'))
        ])
    
  # Create classifier pipeline
  clf = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('classifier', run.model)
  ])
  return clf


def subsample_train_test_split(X, y, max_length=None, test_size=0.2, random_state=None, **kwargs):
    """
    Wrapper for train_test_split that ensures no split exceeds max_length.
    
    Parameters:
    -----------
    X : array-like
        Features data
    y : array-like
        Target data
    max_length : int or None
        Maximum allowed length for any split
    test_size : float or int
        Size of test set (passed to train_test_split)
    random_state : int or None
        Random seed (passed to train_test_split)
    **kwargs : 
        Additional arguments passed to train_test_split
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        The data splits, possibly truncated to max_length
    """
    if max_length is None:
        # If no max_length specified, use standard train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state, **kwargs)
    
    # Calculate how many samples we can use in total
    total_usable = min(len(X), max_length * 2)  # Max we can use while respecting max_length
    
    if total_usable < len(X):
        # Subsample the data before splitting if needed
        indices = np.random.RandomState(random_state).choice(
            len(X), size=total_usable, replace=False)
        X_subset = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_subset = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    else:
        X_subset, y_subset = X, y
    
    # Now split the subsampled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=test_size, random_state=random_state, **kwargs)
    
    # Double-check that each split is below max_length
    # (might be needed if test_size was very small or large)
    if len(X_train) > max_length:
        idx = np.random.RandomState(random_state).choice(
            len(X_train), size=max_length, replace=False)
        X_train = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
        y_train = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
    
    if len(X_test) > max_length:
        idx = np.random.RandomState(random_state).choice(
            len(X_test), size=max_length, replace=False)
        X_test = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
        y_test = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
    
    return X_train, X_test, y_train, y_test
