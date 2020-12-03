import numpy as np
import pandas as pd

def impute_missing_vals(df):
    df['Zip'] = df['Zip'].fillna('Other_Zip').astype(str)
    zips_of_missing_city = df['Zip'][df['City'].isna()]
    df_without_na = df.dropna()
    for zip_code in zips_of_missing_city.unique():
        if zip_code == 'Other_Zip':
            df['City'][df['Zip'] == zip_code] = 'Other_City'
        else:
            df['City'][df['Zip'] == zip_code] = df_without_na['City'][df_without_na['Zip'] == zip_code].values[0]
    df['BusinessType'] = df['BusinessType'].fillna('Other_BT')
    df['NAICSCode'] = df['NAICSCode'].astype(str)
    return df

def preprocess_data(csv_data):
    raw_data = csv_data.copy()
    raw_data = raw_data.drop(columns=['Unnamed: 0'], axis=1)

    data = impute_missing_vals(raw_data.copy())

    data['log_JobsRetained'] = np.log(data['JobsRetained'] + 1)

    upper_quantile = data['log_JobsRetained'].quantile(0.99)
    lower_quantile = data['log_JobsRetained'].quantile(0.01)

    data = data[data['log_JobsRetained'] < upper_quantile]
    data = data[data['log_JobsRetained'] >= lower_quantile]

    data['JobsRetained'] = data['JobsRetained'].astype(int)

    input_features = ['BusinessType', 'CD', 'DateApproved', 'Gender',
           'Lender', 'LoanRange', 'NAICSCode', 'NonProfit', 'RaceEthnicity',
           'State', 'City', 'Zip', 'Veteran']

    sample_data = data.copy()


    sample_data = sample_data.drop_duplicates(input_features, keep='first')
    sample_data = sample_data.sample(frac=1)

    X = sample_data.drop(columns=['JobsRetained', 'log_JobsRetained'], axis=1)[input_features]
    y = sample_data['log_JobsRetained']

    return input_features, X, y