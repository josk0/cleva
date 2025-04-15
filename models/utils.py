"""Methods used by runners"""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import math
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


def bounded_train_test_split(
    X, y,
    max_length=None,
    test_size=0.2,
    random_state=None,
    **kwargs
):
    """
    Wrapper for train_test_split ensuring splits do not exceed max_length
    by calculating the precise number of samples to subset before splitting.
    This avoids the need for potentially stratification-breaking post-split truncation.

    Parameters:
    -----------
    X : array-like
        Features data
    y : array-like
        Target data
    max_length : int or None
        Maximum allowed length for EITHER the train OR the test split.
    test_size : float or int
        Size of the test set.
        If float, should be between 0.0 and 1.0 (exclusive) and represents the proportion.
        If int, represents the absolute number of test samples.
    random_state : int or None
        Random seed for sampling and splitting.
    **kwargs :
        Additional arguments passed to train_test_split (e.g., stratify).

    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        The data splits. Neither split will exceed max_length.

    Raises:
    -------
    ValueError: If constraints cannot be met (e.g., test_size > max_length,
                not enough data for requested test_size, invalid test_size).
    TypeError: If test_size is not float or int.
    """
    if max_length is None:
        # If no max_length, use standard split (no bounding needed)
        print("Using standard train_test_split (max_length is None)")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, **kwargs)

    n_total_original = len(X)
    if n_total_original == 0:
        raise ValueError("Input data X has zero samples.")
    if max_length <= 0:
        raise ValueError(f"max_length ({max_length}) must be positive.")

    # --- Step 1: Calculate N - the maximum total samples to select upfront ---
    n_samples_to_select = n_total_original # Default to using all data initially

    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size as float must be between 0.0 and 1.0 (exclusive)")

        # Constraint 1: Train size N * (1-test_size) <= max_length; thus N <= max_length / (1-test_size)
        max_n_from_train = max_length / (1.0 - test_size)
        # Constraint 2: Test size N * test_size <= max_length; thus N <= max_length / test_size
        max_n_from_test = max_length / test_size

        # Apply constraints: N must be <= original size, <= max N from train, <= max N from test
        n_samples_to_select = math.floor(min(n_total_original, max_n_from_train, max_n_from_test))

    elif isinstance(test_size, int):
        if test_size <= 0:
             raise ValueError("test_size as int must be positive.")
        if test_size > max_length:
             # Requested test set is already larger than allowed max_length for any split
             raise ValueError(f"Requested integer test_size ({test_size}) exceeds max_length ({max_length}).")

        # Constraint 1: Test size is fixed at test_size (which is <= max_length)
        # Constraint 2: Train size N - test_size <= max_length      => N <= max_length + test_size
        max_n_from_train = max_length + test_size

        # Apply constraints: N must be <= original size, <= max N from train constraint
        n_samples_to_select = min(n_total_original, max_n_from_train)

        # Constraint 3: Need enough total samples to even create the test set
        if n_samples_to_select < test_size:
             raise ValueError(
                 f"Cannot satisfy constraints. Need {test_size} samples for the test set, "
                 f"but calculated max usable samples N ({n_samples_to_select}) based on "
                 f"max_length ({max_length}) and available data ({n_total_original}) is too small."
             )
    else:
        raise TypeError(f"test_size must be a float or an int, but got {type(test_size)}")

    # Final check on calculated N
    if n_samples_to_select <= 0:
         raise ValueError(f"Calculated number of samples to select ({n_samples_to_select}) is not positive. Check inputs (max_length, test_size).")


    # --- Step 2: Subsample Data (if N is less than original total) ---
    X_subset, y_subset = X, y
    stratify_arg = kwargs.get('stratify') # Check if stratification is requested

    if n_samples_to_select < n_total_original:
        print(f"Subsampling data: Selecting {n_samples_to_select} out of {n_total_original} samples "
              f"to meet max_length={max_length} constraint with test_size={test_size}.")
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_total_original, size=n_samples_to_select, replace=False) 

        X_subset = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_subset = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    else:
        print(f"Using all {n_total_original} samples (already within constraints for max_length={max_length}).")

    # --- Step 3: Prepare arguments and perform the definitive split ---
    split_kwargs = kwargs.copy()

    # Update stratify argument to use the actual (potentially subsetted) y
    if stratify_arg is not None:
        # Ensure y_subset is valid for stratification
        if len(np.unique(y_subset)) < 2:
             print(f"Warning: Stratification requested but the selected subset has only {len(np.unique(y_subset))} unique classes. Stratification may not be possible or meaningful.")
             # Let train_test_split handle the error if it's strictly impossible (e.g., only 1 class)
        elif len(y_subset) != len(X_subset):
             # This shouldn't happen with correct indexing, but sanity check
             raise ValueError(f"Internal Error: Length mismatch between X_subset ({len(X_subset)}) and y_subset ({len(y_subset)}) before final split.")
        split_kwargs['stratify'] = y_subset


    # Perform the split on the precisely calculated subset
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset,
        test_size=test_size, # Use the original test_size (float or int)
        random_state=random_state, # Use same random state for reproducibility
        **split_kwargs # Pass other args, including corrected stratify
    )

    # --- Step 4: Verification (Optional) ---
    if len(X_train) > max_length:
         print(f"WARNING: Train split ({len(X_train)}) exceeded max_length ({max_length}) unexpectedly! Check calculation logic.")
    if len(X_test) > max_length:
         print(f"WARNING: Test split ({len(X_test)}) exceeded max_length ({max_length}) unexpectedly! Check calculation logic.")
    # If test_size was int, verify the test set size exactly matches (train_test_split should ensure this)
    if isinstance(test_size, int) and len(X_test) != test_size:
         print(f"WARNING: Test split size ({len(X_test)}) doesn't match requested integer test_size ({test_size})!")


    return X_train, X_test, y_train, y_test