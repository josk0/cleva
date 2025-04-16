"""Methods that load datasets, perform basic cleaning, and return frame or X,y"""

import os
import pandas as pd
import numpy as np
from .utils import clean_text_columns, replace_by_dictionary, keep_only_columns, get_dataset_from_kaggle, get_dataset_from_UCI
import data.constants as constants
from functools import cache

@cache
def load_credit_default(as_frame=False):
    """ Load Taiwanese Credit Deafult data """
    # data source: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
    file_path = './data/raw/default of credit card clients.xls'

    if not os.path.exists(file_path):
        print("No local copy of dataset found. Loading copy from UCI ML Repo..")
        get_dataset_from_UCI(url='https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip', dest_file=file_path)
        
    if os.path.exists(file_path):
        data = pd.read_excel(file_path, index_col=0, header=1)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Commented out but kept for reference
    # categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month']
    # data[categorical_columns] = data[categorical_columns].astype('category') 

    if as_frame:
        return data
    else:
        X = data.loc[:, data.columns != 'default payment next month']
        y = data['default payment next month']
    return X, y

@cache
def load_us_perm_visas(as_frame=False, shuffle_data=True, random_state=42):
    """ Load US Permanent Visa Applications data """
    # data source: https://www.kaggle.com/datasets/jboysen/us-perm-visas

    # Check if the file exists
    file_path = './data/raw/us_perm_visas.csv'
    if not os.path.exists(file_path):
        print("No local copy of dataset found. Loading copy from Kaggle..")
        get_dataset_from_kaggle(dataset_name="jboysen/us-perm-visas", file_name="us_perm_visas.csv")
    
    if os.path.exists(file_path):
        # Load the file if it exists
        data = pd.read_csv(file_path, low_memory=False)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    columns_to_keep = ["case_status", "decision_date", "employer_name", "employer_city",
            "employer_state", "job_info_work_city", "job_info_work_state", "pw_soc_code", 
            "pw_unit_of_pay_9089", "pw_source_name_9089", "pw_soc_title",               
            "country_of_citizenship", "class_of_admission", "pw_level_9089",
            "pw_amount_9089"]

    data = keep_only_columns(data, columns_to_keep)

    # Removing all withdrawn applications
    data = data[data.case_status != 'Withdrawn']

    # Remove all rows where both employer_name AND employer_city are missing
    data = data.dropna(subset=['employer_name', 'employer_city'], how='all')

    # Combine certified-expired and certified applications and displaying distribution of "case_status" variable
    data.loc[data.case_status == 'Certified-Expired', 'case_status'] = 'Certified'

    # Strip and lowercase all text columns
    data = clean_text_columns(data)

    # Transform label from string to binary 
    #   Some scoring functions require clear indentification of positive class
    #   We opt into no downcasting and set the int type explicitly
    pd.set_option('future.no_silent_downcasting', True)
    data['case_status'] = data['case_status'].replace({'certified': 1, 'denied': 0})
    data['case_status'] = data['case_status'].astype(int)

    # Standardize entries in some columns 
    data = replace_by_dictionary(data, constants.state_name_to_code_map, ['employer_state', 'job_info_work_state']) # Standardize state codes
    data = replace_by_dictionary(data, constants.time_period_to_abbreviation_map, ['pw_unit_of_pay_9089']) # Standardize time periods

    datetime_columns = ['decision_date']
    categorical_columns = ['employer_state', 
                            'job_info_work_state',
                            'pw_unit_of_pay_9089',
                            'class_of_admission',
                            'pw_source_name_9089',
                            'pw_level_9089'
                            ]
    numerical_columns = ['pw_amount_9089']

    data[datetime_columns] = data[datetime_columns].apply(pd.to_datetime)
    data[categorical_columns] = data[categorical_columns].astype('category') 

    # Deal with numerical columns
    # 1. Remove commas
    data[numerical_columns] = data[numerical_columns].replace(',', '', regex=True)
    
    # 2. Better handling of NA values and infinities
    for col in numerical_columns:
        # Replace problematic strings with NaN
        data[col] = data[col].replace(['NA', 'inf', '-inf', ''], np.nan)
        # Convert to float
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Reset index to make it sequential again (after dropping rows)
    data = data.reset_index(drop=True)

    # Shuffle to avoid that all the missing values are clustered together
    #   this avoids a problem specifically with 'country_of_citizenship'
    #   in which all missing values are in the first 19k rows
    #   which, in turn, creates a probem for imputers in the pipeline
    if shuffle_data:
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    

    if as_frame:
        return data
    else:
        X = data.loc[:, data.columns != "case_status"]
        y = data['case_status']
        return X, y