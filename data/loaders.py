"""Methods that load datasets, perform basic cleaning, and return frame or X,y"""

import pandas as pd
from .cleaning import clean_text_columns, replace_by_dictionary, keep_only_columns
import data.constants as constants
from functools import cache

@cache
def load_credit_default(as_frame=False):
    """ Load Taiwanese Credit Deafult data """
    # data source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    data = pd.read_excel('./data/raw/default of credit card clients.xls', index_col=0, header=1)

    # categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month']
    # data[categorical_columns] = data[categorical_columns].astype('category') 

    if as_frame:
        return data
    else:
        X = data.loc[:, data.columns != 'default payment next month']
        y = data['default payment next month']
    return X, y

@cache
def load_us_perm_visas(as_frame=False):
    data = pd.read_csv('./data/raw/us_perm_visas.csv', low_memory=False)

    columns_to_keep = ["case_status","decision_date","employer_name","employer_city",
            "employer_state","job_info_work_city","job_info_work_state","pw_soc_code"                
            ,"pw_unit_of_pay_9089","pw_source_name_9089","pw_soc_title"               
            ,"country_of_citizenship","class_of_admission" ,"pw_level_9089",
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
    
    # 2. Remove missing and infitite values (TabPFN doesn't like these)
    data[numerical_columns] = data[numerical_columns].replace('NA', '', regex=True)
    data[numerical_columns] = data[numerical_columns].replace('inf', '', regex=True)
    
    # 3. Convert to float (to handle decimal points)
    data[numerical_columns] = data[numerical_columns].astype(float) # not sure: could also transform to int64 (after np.floor), since most values don't have decimals

    if as_frame:
        return data
    else:
        X = data.loc[:, data.columns != "case_status"]
        y = data['case_status']
        return X, y