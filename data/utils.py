"""Methods used by loaders to do basic data loading cleaning"""

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

def get_dataset_from_kaggle(dataset_name: str, file_name: str):
    import kagglehub
    import os
    import shutil

    # Download latest version
    path = kagglehub.dataset_download(dataset_name)
    # print("Path to dataset files:", path)
    source_file = os.path.join(path, file_name)

    # Create destination directory if it doesn't exist
    dest_dir = os.path.join("data", "raw")
    os.makedirs(dest_dir, exist_ok=True)

    # Create full destination path
    dest_file = os.path.join(dest_dir, file_name)

    # Copy the file
    shutil.copy2(source_file, dest_file)

    print(f"Copied {source_file} to {dest_file}")

def get_dataset_from_UCI(url: str, dest_file: str):
    import urllib.request
    import zipfile
    import tempfile
    import shutil
    import os
    
   # Create destination directory if it doesn't exist
    dest_dir = os.path.join("data", "raw")
    os.makedirs(dest_dir, exist_ok=True)
    
    # Download to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        urllib.request.urlretrieve(url, temp_file.name)
        temp_path = temp_file.name
    
    # Extract the zip file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find and copy the extracted file to the target location
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.xls'):
                    source = os.path.join(root, file)
                    shutil.copy(source, dest_file)
                    break
    
    # Clean up the temporary zip file
    os.unlink(temp_path)