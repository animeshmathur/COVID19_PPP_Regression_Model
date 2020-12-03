import numpy as np
import pandas as pd

from PPP_Data_Preprocessing import preprocess_data, impute_missing_vals

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

import pickle

class RMSLEMetric(object):
    ''' Custom RMSLE Metric for Catboost'''
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            pred = approx[i]
            if pred < 0:
                pred = 0
            error_sum += w * ((pred - target[i])**2)

        return error_sum, weight_sum

csv_data = pd.read_csv('./2020 PPP Dataset/PPP Train ALL.csv')

input_features, X, y = preprocess_data(csv_data)

categorical_features_indices = np.where(X.dtypes != np.float)[0]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostRegressor(iterations=1,
                                  depth=10,
                                  learning_rate=0.1,
                                  eval_metric=RMSLEMetric(),
                                  objective='RMSE',
                                  use_best_model=True,
                                  thread_count=6,
                                  early_stopping_rounds=1)

model.fit(X_train_f, 
          y_train_f,
          cat_features=categorical_features_indices,
          eval_set=(X_test_f, y_test_f),
          plot=True)

y_pred = model.predict(X)
y_pred = np.exp(y_pred) - 1
y_pred[y_pred < 0] = 0
print("Training RMSLE score: ", np.sqrt(mean_squared_log_error(np.exp(y) - 1, y_pred)))

test_data = pd.read_csv('./2020 PPP Dataset/PPP Test ALL.csv')

test_data = impute_missing_vals(test_data)

test_pred = model.predict(test_data[input_features])
test_pred = np.exp(test_pred) - 1
test_pred[test_pred < 0] = 0

submission = pd.DataFrame(columns=['Index', 'JobsRetained'])
submission['Index'] = test_data['Index'].copy()
submission['JobsRetained'] = np.round(test_pred, 0).astype(int)

export_model = open('./models/catboost_regressor.pkl', 'wb')
pickle.dump(model, export_model)
export_model.close()

submission.set_index('Index',inplace=True)
submission.to_csv('./Submission.csv')