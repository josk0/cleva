"""Methods used by loaders to do basic data cleaning"""

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
