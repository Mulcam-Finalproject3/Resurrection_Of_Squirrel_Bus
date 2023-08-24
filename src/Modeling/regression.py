import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = './Data/csv'

def get_parquet_data(file_path):
    import pandas as pd

    ride_df = pd.read_parquet(f"{file_path}/ride_df/")
    alight_df = pd.read_parquet(f"{file_path}/alight_df/")

    ride_df['lon'] = ride_df['lon'].astype('double')
    ride_df['lat'] = ride_df['lat'].astype('double')

    alight_df['lon'] = alight_df['lon'].astype('double')
    alight_df['lat'] = alight_df['lat'].astype('double')

    ride_df = ride_df.drop(["NODE_ID"], axis = 1)
    alight_df = alight_df.drop(["NODE_ID"], axis = 1)

    return ride_df, alight_df

# from sklearn.base import BaseEstimator, TransformerMixin

# class regression_estimator(BaseEstimator, TransformerMixin):
#     pass

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
        dataframe[target_data],
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

    rf_score = pd.DataFrame(grid_rf.cv_results_)
    df_rf_score = rf_score[['params','mean_test_score','rank_test_score',
            'split0_test_score','split1_test_score','split2_test_score','split3_test_score','split4_test_score']]

    print(df_rf_score)
    print('GridSearchCV 최적 파라미터 :', grid_rf.best_params_)
    print('GridSearchCV 최적 파라미터의 평균 MSE :{:.3f}'.format(grid_rf.best_score_))

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
        return_train_score = True, # 훈련 세트에 대한 평가 점수도 같이 받게 해주는 옵션
        n_jobs = -1, # 사용 가능한 CPU 코어를 모두 사용해서 훈련에 투입
        cv = 3 # 각 하이퍼 파라미터 조합으로 만드는 모델에서 사용할 폴드의 개수
        # parameters의 경우의 수 6 x 폴드의 개수 3 = 총 18개의 모델 생성
    )

    grid_xgb.fit(X_train, y_train)

    print("GridSearchCV의 최적 하이퍼 파라미터 : {}".format(grid_xgb.best_params_))
    print("GridSearchCV의 최고 정확도 : {: .4f}".format(grid_xgb.best_score_))
    
    best_model = grid_xgb.best_estimator_
    return best_model

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
