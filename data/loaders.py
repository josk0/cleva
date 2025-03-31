import pandas as pd
from .cleaning import clean_text_columns, replace_by_dictionary, prepare_for_tabpfn
import constants

def load_us_perm_visas(as_frame=False):
  data = pd.read_csv('./data/raw/us_perm_visas.csv', low_memory=False)

  # Removing all withdrawn applications
  data = data[data.case_status != 'Withdrawn']

  # Combining certified-expired and certified applications and displaying distribution of "case_status" variable
  data.loc[data.case_status == 'Certified-Expired', 'case_status'] = 'Certified'

  # stip and lowercase all text columns
  data = clean_text_columns(data)

  # Standardize entries in some columns 
  data = replace_by_dictionary(data, constants.state_name_to_code_map, ['employer_state', 'job_info_work_state']) # Standardize state codes
  data = replace_by_dictionary(data, constants.time_period_to_abbreviation_map, ['pw_unit_of_pay_9089', 'wage_offer_unit_of_pay_9089']) # Standardize time periods

  datetime_columns = ['decision_date']
  categorical_columns = ['employer_state', 
                         'job_info_work_state',
                         'pw_unit_of_pay_9089',
                         'class_of_admission',
                         'pw_source_name_9089',
                         'pw_level_9089',
                         'wage_offer_unit_of_pay_9089'
                         ]

  data[datetime_columns] = data[datetime_columns].apply(pd.to_datetime)
  data[categorical_columns] = data[categorical_columns].astype('category') 

  # Handle missing values
  data = prepare_for_tabpfn(data)

  # Create X and y
  # X = data.drop('case_status', axis=1)
  # X = data[["decision_date","employer_name","employer_city",
  #         "employer_state","job_info_work_city","job_info_work_state","pw_soc_code"                
  #         ,"pw_unit_of_pay_9089","pw_source_name_9089","pw_soc_title"               
  #         ,"country_of_citizenship","class_of_admission" ,"pw_level_9089",
  #         "pw_amount_9089","wage_offer_unit_of_pay_9089"]]

  # Because we called prepare_for_tabpfn, which changed the datetime columns, we need to have a different selection of columns
  X = data[["decision_date_year", "decision_date_month", "employer_name","employer_city",
          "employer_state","job_info_work_city","job_info_work_state","pw_soc_code"                
          ,"pw_unit_of_pay_9089","pw_source_name_9089","pw_soc_title"               
          ,"country_of_citizenship","class_of_admission" ,"pw_level_9089",
          "pw_amount_9089","wage_offer_unit_of_pay_9089"]]


  y = data['case_status']

  return X, y