import numpy as np
import pandas as pd

from PPP_Data_Preprocessing import impute_missing_vals

import pickle

input_features = ['BusinessType', 'CD', 'DateApproved', 'Gender',
           'Lender', 'LoanRange', 'NAICSCode', 'NonProfit', 'RaceEthnicity',
           'State', 'City', 'Zip', 'Veteran']

model_pkl = open('./catboost_regressor.pkl', 'rb')
model = pickle.load(model_pkl)
model_pkl.close()

test_data = pd.read_csv('./2020 PPP Dataset/PPP Test ALL.csv')

test_data = impute_missing_vals(test_data)

test_pred = model.predict(test_data[input_features])
test_pred = np.exp(test_pred) - 1
test_pred[test_pred < 0] = 0

submission = pd.DataFrame(columns=['Index', 'JobsRetained'])
submission['Index'] = test_data['Index'].copy()
submission['JobsRetained'] = np.round(test_pred, 0).astype(int)

submission.set_index('Index',inplace=True)
submission.to_csv('./Submission.csv')