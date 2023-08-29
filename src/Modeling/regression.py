import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = './Data/csv'

def get_parquet_data(file_path):
    import pandas as pd

    ride_df = pd.read_parquet(f"{file_path}/ride_df_agg/")
    alight_df = pd.read_parquet(f"{file_path}/alight_df_agg/")

    ride_df['lon'] = ride_df['lon'].astype('double')
    ride_df['lat'] = ride_df['lat'].astype('double')

    alight_df['lon'] = alight_df['lon'].astype('double')
    alight_df['lat'] = alight_df['lat'].astype('double')

    ride_df = ride_df.drop(["NODE_ID"], axis = 1)
    alight_df = alight_df.drop(["NODE_ID"], axis = 1)

    return ride_df, alight_df

def log_transformation(dataframe, target_data):
    import numpy as np

    dataframe[f'{target_data}_log'] = np.log1p(dataframe[target_data])

    return dataframe

def encoding_data(dataframe):
    import pandas as pd

    from sklearn.preprocessing import StandardScaler

    ohe_df = pd.get_dummies(dataframe['Hour'], columns=['Hour'])

    standard_scaler = StandardScaler()

    scale_cols = ["lon", "lat", "tmp", "wsd", "pcp"]
    standard_df = dataframe[scale_cols]
    standard_df = pd.DataFrame(standard_scaler.fit_transform(standard_df), 
                               columns=standard_df.columns)

    concat_df = pd.concat([standard_df, ohe_df], axis = 1)

    return concat_df

def split_data(dataframe, target_data):
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe,
        target_data,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def randomforest_grid_search(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    rf_model = RandomForestRegressor(n_jobs=-1,random_state=42)

    grid_parameters ={
        'n_estimators':[100,200],
        'max_depth':[5,10,15],
        'max_features':[1, 0.9, 0.8],
        'min_samples_leaf': [2, 4, 6]
    }

    grid_rf = GridSearchCV(rf_model, param_grid=grid_parameters, scoring='neg_mean_squared_error', cv=5, refit=True)
    grid_rf.fit(X_train, y_train)

    best_model = grid_rf.best_estimator_

    return best_model

def XGB_grid_search(X_train, y_train):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    parameters = {
        "max_depth": [1, 2, 3],
        "learning_rate": [0.05, 0.1, 0.3, 0.5],
        "n_estimators": [100, 200, 300, 400],
        "subsample": [0.1, 0.3, 0.5],
        "colsample_bytree": [0.1, 0.3, 0.5]
    }

    grid_xgb = GridSearchCV(
        XGBRegressor(n_estimators = 100, subsample = 0.5),
        param_grid = parameters,
        return_train_score = True,
        n_jobs = -1,
        cv = 3
        )

    grid_xgb.fit(X_train, y_train)

    best_model = grid_xgb.best_estimator_

    return best_model

def model_eval(best_model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import mean_squared_error,r2_score

    best_model.fit(X_train, y_train)

    y_pred_train = best_model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    y_pred_test = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    train_rmse = np.sqrt(abs(mse_train))
    test_rmse = np.sqrt(abs(mse_test))

    print(f"{best_model} RMSE (train) : ", train_rmse)
    print(f"{best_model} R2 (train) : ", r2_train)
    print(f"{best_model} RMSE (test) : ", test_rmse)
    print(f"{best_model} R2 (test) : ", r2_test)

def save_model(best_model):
    import joblib
    joblib.dump(best_model, f'xgb_{best_model}.pkl')


def load_model(model):
    import joblib

    model = joblib.load(f'{model}.pkl')
    return model

def get_weather_data(file_path, file_name):
    import pandas as pd

    pred_weather_df = pd.read_csv(f'{file_path}/{file_name}', encoding = 'euc-kr')
    pred_weather_df.drop(['VEC', 'REH', 'X', 'Y', 'Date'], axis = 1, inplace=True)
    pred_weather_df.rename(columns={'raw_x':'lon', 'raw_y': 'lat', 'time': 'Hour', 'TMP': 'TMP'.lower(), 'WSD': 'WSD'.lower(), 'PCP': 'PCP'.lower()}, inplace=True)
    pred_weather_df.loc[(pred_weather_df['Hour'] == 600)| 
                        (pred_weather_df['Hour'] == 700) | 
                        (pred_weather_df['Hour'] == 800) | 
                        (pred_weather_df['Hour'] == 900), 'Hour'] = pred_weather_df['Hour'] // 100
    
    return pred_weather_df

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

class RegressionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target, rename_cols = None):
        self.target = target
        self.rename_cols = rename_cols
        self.drop_cols = ['VEC', 'REH', 'X', 'Y', 'Date']
        self.scaled_cols = ["lon", "lat", "tmp", "wsd", "pcp"]
        self.encoded_cols = ['Hour']

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()  
    def fit(self):

        return self

    def fit_transform(self, dataframe):
        # 로그 변환 적용
        dataframe[self.target] = np.log1p(dataframe[self.target])
        # 컬럼명 변경
        if self.rename_cols:
            dataframe = dataframe.rename(columns=self.rename_cols)
        
        if (dataframe['Hour'] == 600 | ['Hour'] == 700 | ['Hour'] == 800 | ['Hour'] == 900):
            dataframe.loc[(dataframe['Hour'] == 600) | 
                          (dataframe['Hour'] == 700) | 
                          (dataframe['Hour'] == 800) | 
                          (dataframe['Hour'] == 900), 'Hour'] = dataframe['Hour'] // 100
    
        ohe_df = pd.get_dummies(dataframe['Hour'], columns=['Hour'])

        standard_df = dataframe[self.scaled_cols]
        standard_df = pd.DataFrame(self.scaler.fit_transform(standard_df), 
                                   columns=standard_df.columns)

        concat_df = pd.concat([standard_df, ohe_df], axis = 1)

        return concat_df
      

# class RandomForestRegressionPipeline():
#     def __init__(self):

#     self.steps = [
#         ('scaler', self.scaler),
#         ('encoder', self.encoder),
#         ('log_transform', self.target)
#         ('rf_model', RandomForestRegressor(self.best_model.get_params()))
#     ]

#     # 파이프라인 객체 생성
#     self.pipeline = Pipeline(self.steps)

#     def fit():
#         pass

#     def predict():
#         pass
