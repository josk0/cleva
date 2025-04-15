"""Methods used by runners"""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
from models.preprocessors import DatetimeFeatureSplitter, DatetimeFeatureEncoder


def get_pipeline(run):
    if run is None:
        raise ValueError("Run must be provided.")
  
    # Create preprocessor based on model
    if "RandomForest" in run.model_name:
        print("RandomForest model detected.")
        transformers=[
            ('datetime', DatetimeFeatureEncoder(), make_column_selector(dtype_include='datetime64[ns]')),
            ('cat high c', TargetEncoder(), make_column_selector(dtype_include='object')),
            ('num',SimpleImputer(strategy='constant', fill_value=0), make_column_selector(dtype_include=['int64', 'float64'])), # no need to scale numerical columns for RandomForest
            ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include='category'))
            # ('pass', 'passthrough', make_column_selector(dtype_exclude=['datetime64']))
        ]
    elif "CatBoost" in run.model_name:
        print("CatBoost model detected.")
        transformers=[
            ('datetime', DatetimeFeatureEncoder(), make_column_selector(dtype_include='datetime64[ns]')),
            ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            ('pass', 'passthrough', make_column_selector(dtype_exclude=['datetime64', 'int64', 'float64']))
        ]
    elif "TabPFN" in run.model_name:
        print("TabPFN model detected.")
        transformers=[
            ('datetime', DatetimeFeatureSplitter(), make_column_selector(dtype_include='datetime64[ns]')),
            ('num', SimpleImputer(strategy='mean'), make_column_selector(dtype_include=['int64', 'float64'])), # Here we see the limit of the approach: different strategy between models
            ('cat', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_include='category')),
            ('cat high cardinality', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_include='object')),
            # todo: this isn't great, having to impute values for TabPFN. But the model otherwise had issues with missing variables. 
            # I think TabPFN expects missing values to be formatted in a certain way. Here it got an NA type or so in what it expected to be a str column

        ]
    else:
        # Default preprocessing for unrecognized models
        print(f"Using default preprocessing for {run.model_name}")
        transformers=[
            ('datetime', DatetimeFeatureEncoder(), make_column_selector(dtype_include='datetime64[ns]')),
            ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include='category')),
            ('cat high c', TargetEncoder(), make_column_selector(dtype_include='object'))
        ]

    # Create preprocessor
    if any(transformers):
        preprocessor = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )
        # Create classifier pipeline
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', run.model)
        ])
    else: # If no preprocessors, pipeline is just the model itself
        clf = run.model
        
    return clf


def bounded_train_test_split(X, y, max_length=None, test_size=0.2, random_state=None, **kwargs):
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
        Additional arguments passed to train_test_split (e.g., stratify)
        
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

    X_subset, y_subset = X, y # Initialize with full data
    stratify_arg = kwargs.get('stratify') # Get potential stratify argument

    if total_usable < len(X):
        # Subsample the data before splitting if needed
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(X), size=total_usable, replace=False)

        # Use .iloc for pandas, slicing for numpy
        X_subset = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        # Ensure y_subset matches X_subset
        y_subset = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]

    # --- Prepare arguments for train_test_split ---
    split_kwargs = kwargs.copy() # Avoid modifying original kwargs dict

    # *** FIX for Stratify ***
    # If stratification was requested, update the stratify argument
    # to use the actual subset of y that corresponds to X_subset.
    if stratify_arg is not None:
         # Check if y_subset has the same length as X_subset, if not, maybe raise error or warning
        if len(y_subset) != len(X_subset):
             raise ValueError(f"Internal Error: Length mismatch between X_subset ({len(X_subset)}) and y_subset ({len(y_subset)}) after subsampling.")
        split_kwargs['stratify'] = y_subset
    # *************************

    # Now split the (potentially subsampled) data
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset,
        test_size=test_size,
        random_state=random_state,
        **split_kwargs # Pass potentially modified kwargs (with corrected stratify)
    )

    # --- Post-Split Truncation (Original logic - with potential stratification issue) ---
    # This truncation happens *after* the split and can break stratification.
    # Consider if this truncation is acceptable or if the initial sampling
    # should be more precise to avoid this step.
    rng_trunc = np.random.RandomState(random_state) # Use same seed for deterministic truncation

    if len(X_train) > max_length:
        print(f"Warning: Training set ({len(X_train)}) exceeded max_length ({max_length}) after split. Truncating randomly.")
        idx_train = rng_trunc.choice(len(X_train), size=max_length, replace=False)
        X_train = X_train.iloc[idx_train] if hasattr(X_train, 'iloc') else X_train[idx_train]
        y_train = y_train.iloc[idx_train] if hasattr(y_train, 'iloc') else y_train[idx_train]

    if len(X_test) > max_length:
        print(f"Warning: Test set ({len(X_test)}) exceeded max_length ({max_length}) after split. Truncating randomly.")
        idx_test = rng_trunc.choice(len(X_test), size=max_length, replace=False)
        X_test = X_test.iloc[idx_test] if hasattr(X_test, 'iloc') else X_test[idx_test]
        y_test = y_test.iloc[idx_test] if hasattr(y_test, 'iloc') else y_test[idx_test]

    return X_train, X_test, y_train, y_test