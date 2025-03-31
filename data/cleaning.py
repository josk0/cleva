import pandas as pd
import yaml
import numpy as np

# Usage in your cleaning script:
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def clean_dataframe(df, config):
    df = df.copy()
    
    # Drop specified columns
    df = df.drop(columns=config['columns']['drop'], errors='ignore')
    
    # Convert datetime columns
    for col in config['columns']['datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], 
                                   format=config['cleaning_rules']['datetime_format'],
                                   errors='coerce')
    
    # Apply string cleaning rules
    if config['cleaning_rules']['string_columns']['strip_whitespace']:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
    
    # Handle missing values
    for col in df.select_dtypes(include=['number']).columns:
        if config['missing_values']['numerical']['method'] == 'mean':
            df[col] = df[col].fillna(df[col].mean())
    
    return df

# # Using the configuration:
# config = load_config()
# clean_df = clean_dataframe(df, config)

def keep_only_columns(df, to_keep):
    """
    Keep only specified columns in a dataframe.
    
    Args:
        df (pandas.DataFrame): The input dataframe
        to_keep (list): List of column names to keep
        
    Returns:
        pandas.DataFrame: Dataframe with only the specified columns
    """
    # Filter to only include columns in to_keep that exist in df
    valid_cols = [col for col in to_keep if col in df.columns]
    
    # Return dataframe with only those columns
    return df[valid_cols]

def drop_empty_rows_and_columns(df):
  # Dropping all empty columns
  df = df.dropna(axis=1, how='all')

  # Dropping all empty rows
  df = df.dropna(axis=0, how='all')

def clean_text_columns(df):
    """
    Clean all string columns in a DataFrame by:
    1. Stripping leading/trailing whitespace
    2. Converting text to lowercase
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to process
    
    Returns:
    pandas.DataFrame: Processed DataFrame with cleaned text columns
    """
    # Get all object dtype columns
    string_columns = df.select_dtypes(include=['object']).columns
    
    # Create a copy containing only the string columns
    cleaned_df = df.copy()
    
    # Use pandas' apply with a lambda function to perform operations in one pass
    if len(string_columns) > 0:
        cleaned_df[string_columns] = cleaned_df[string_columns].apply(
            lambda x: x.str.strip().str.lower() if x.dtype == 'object' else x
        )
    
    return cleaned_df

def replace_by_dictionary(df, replacement_dict, columns):
    """
    Replace values in a DataFrame column based on a dictionary mapping.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the state column(s)
    replacement_dict (dict): Mapping from stings to replacements
    columns (str or list): Column name or list of column names containing state information
    
    Returns:
    pandas.DataFrame: DataFrame with standardized state codes
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Handle both single column name and list of column names
    if isinstance(columns, str):
        columns = [columns]
    
    # Apply the mapping to convert state names to codes for each column
    for column in columns:
        df_copy[column] = df_copy[column].apply(
            lambda x: replacement_dict.get(x, x) if isinstance(x, str) else x
        )
    
    return df_copy

def prepare_for_tabpfn(df):
    """
    Prepare a DataFrame for TabPFN by:
    1. Converting datetime columns to numeric features
    2. Handling missing values properly
    3. Ensuring all features are compatible
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to process
    
    Returns:
    pandas.DataFrame: TabPFN-ready DataFrame
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Step 1: Handle datetime columns by converting to numeric
    for col in df_copy.select_dtypes(include=['datetime64']).columns:
        # Extract multiple numeric features from datetime
        df_copy[f"{col}_year"] = df_copy[col].dt.year
        df_copy[f"{col}_month"] = df_copy[col].dt.month
        df_copy[f"{col}_day"] = df_copy[col].dt.day
        df_copy[f"{col}_dayofweek"] = df_copy[col].dt.dayofweek
        
        # Drop the original datetime column
        df_copy = df_copy.drop(columns=[col])
    
    # Step 2: Convert DataFrame to numpy array and back to ensure pandas NA conversion
    array_data = df_copy.to_numpy(dtype=object)
    df_copy = pd.DataFrame(array_data, columns=df_copy.columns, index=df_copy.index)
    
    # Step 3: Handle string/object columns
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Replace None and NaN with a string placeholder
        mask = df_copy[col].isna()
        if mask.any():
            df_copy.loc[mask, col] = "missing"
    
    # Step 4: Handle numeric columns
    for col in df_copy.select_dtypes(include=['number']).columns:
        # Ensure NaN for missing values in numeric columns
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Step 5: Replace infinities with np.nan
    pd.set_option('future.no_silent_downcasting', True) # to avoid FutureWarning about implicit downcasting in .replace
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
    
    # Step 6: Final check for any remaining pandas NA values
    for col in df_copy.columns:
        # For object columns, replace NA with string "missing"
        if pd.api.types.is_object_dtype(df_copy[col].dtype):
            df_copy[col] = df_copy[col].fillna("missing")
        # For numeric columns, impute with mean (or other strategy of your choice)
        elif pd.api.types.is_numeric_dtype(df_copy[col].dtype):
            # Replace NA with column mean, or 0 if the column is all NA
            if df_copy[col].isna().all():
                df_copy[col] = df_copy[col].fillna(0)
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    return df_copy

def consistent_missing_values(df):
    """
    Ensure consistent handling of missing values for compatibility with TabPFN and sklearn.
    Forcefully converts all pandas NA and any other non-standard NA representations to formats
    that sklearn encoders can handle.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to process
    
    Returns:
    pandas.DataFrame: DataFrame with consistent missing value representation
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert DataFrame to numpy array and then back to DataFrame
    # This forces pandas NA values to be converted to Python None or np.nan
    array_data = df_copy.to_numpy()
    df_copy = pd.DataFrame(array_data, columns=df_copy.columns, index=df_copy.index)
    
    # For object/string columns, explicitly convert None to string "missing"
    # This ensures uniform typing for string columns
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Replace None and NaN with a string placeholder
        mask = df_copy[col].isna()
        if mask.any():
            df_copy.loc[mask, col] = "missing"
    
    # For numeric columns, ensure np.nan is used (not None or other types)
    for col in df_copy.select_dtypes(include=['number']).columns:
        # Use pandas replace to ensure NaN for any missing numeric values
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Replace infinities with np.nan
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
    
    # Double-check for any remaining pandas NA values and forcefully convert them
    for col in df_copy.columns:
        # Check if column contains pandas NA values
        if df_copy[col].isna().any():
            # Get the dtype of the column
            col_dtype = df_copy[col].dtype
            
            # For object columns, replace NA with string "missing"
            if pd.api.types.is_object_dtype(col_dtype):
                df_copy[col] = df_copy[col].fillna("missing")
            # For numeric columns, replace NA with np.nan
            elif pd.api.types.is_numeric_dtype(col_dtype):
                df_copy[col] = df_copy[col].fillna(np.nan)
    
    return df_copy

def consistent_missing_values_old(df):
    """
    Ensure consistent handling of missing values for compatibility with TabPFN and sklearn.
    Converts pandas NA to None/np.nan which sklearn can handle better.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to process
    
    Returns:
    pandas.DataFrame: DataFrame with consistent missing value representation
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # For all columns, replace pandas NA with Python None
    # This is crucial for compatibility with sklearn encoders
    for col in df_copy.columns:
        # Replace pandas NA with None (which sklearn can better handle)
        df_copy[col] = df_copy[col].replace(pd.NA, None)
    
    # For string columns, missing values can be represented as a string if needed
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Option 1: Convert None to "MISSING" if you want a string placeholder
        # df_copy[col] = df_copy[col].fillna("MISSING")
        
        # Option 2: Keep as None (better for TabPFN compatibility)
        pass
    
    # For numeric columns, use np.nan
    for col in df_copy.select_dtypes(include=['number']).columns:
        df_copy[col] = df_copy[col].fillna(np.nan)
    
    # Replace infinities with np.nan
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

    return df_copy