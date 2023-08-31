import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

path = './Data/csv'

def get_parquet_data(file_path):
    """parquet 형식 데이터 pandas dataframe으로 불러오기

    Args:
        file_path (String): 파일 경로

    Returns:
        ride_df, alight_df: 승차 dataframe, 하차 dataframe
    """
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

class RegressionTransformer(BaseEstimator, TransformerMixin):
    """회귀 예측을 위한 데이터 변환기

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def __init__(self, target = None, rename_cols = None, drop_cols = None):
        self.target = target
        self.rename_cols = rename_cols
        self.drop_cols = drop_cols
        self.scaled_cols = ["lon", "lat", "tmp", "wsd", "pcp"]
        self.encoded_cols = ['Hour']

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        self.rf_model = RandomForestRegressor()
        
        # self.steps = [
        #     ('scaler', self.scaler),
        #     ('encoder', self.encoder),
        #     ('rf_model', self.rf_model)
        # ]

        # # 파이프라인 객체 생성
        # self.pipeline = Pipeline(self.steps)

    def fit(self, dataframe):
        # 로그 변환 적용
        if self.target:
            dataframe[self.target] = np.log1p(dataframe[self.target])
        
        # 컬럼명 변경
        if self.rename_cols:
            dataframe = dataframe.rename(columns=self.rename_cols)

        # 컬럼 삭제
        if self.drop_cols:
            dataframe = dataframe.drop(columns=self.drop_cols, axis = 1)
        
        # 시간대 데이터를 형식에 맞게 변경
        dataframe.loc[(dataframe['Hour'] == 600)| 
                (dataframe['Hour'] == 700) | 
                (dataframe['Hour'] == 800) | 
                (dataframe['Hour'] == 900), 'Hour'] = dataframe['Hour'] // 100
        
        return dataframe

    def transform(self, dataframe):
        # 원핫 인코딩
        ohe_df = pd.get_dummies(dataframe['Hour'], columns=['Hour'])
        ohe_df = ohe_df.rename(columns={ohe_df.columns[0]: 'six', ohe_df.columns[1]: 'seven', ohe_df.columns[2]: 'eight', ohe_df.columns[3]: 'nine'})

        # 스케일러 적용
        standard_df = dataframe[self.scaled_cols]
        standard_df = pd.DataFrame(self.scaler.fit_transform(standard_df), 
                                   columns=standard_df.columns)
        
        # 데이터프레임 결합
        concat_df = pd.concat([standard_df, ohe_df], axis = 1)

        return concat_df
    
def randomforest_grid_search(X_train, y_train):
    """Randomforest regression GridSearchCV, 최적 모델 pkl 파일로 저장

    Args:
        X_train (_type_): Feature
        y_train (_type_): target

    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    rf_model = RandomForestRegressor(n_jobs=-1,random_state=42)

    grid_parameters ={
        'n_estimators':[100, 200],
        'max_depth':[5, 10, 15],
        'max_features':[1, 0.9, 0.8],
        'min_samples_leaf': [2, 4, 6]
    }

    grid_rf = GridSearchCV(rf_model, 
                           param_grid = grid_parameters, 
                           scoring='neg_mean_squared_error', 
                           cv=5, 
                           refit=True)
    grid_rf.fit(X_train, y_train)

    best_model = grid_rf.best_estimator_

    return best_model

def XGB_grid_search(X_train, y_train):
    """XGB GridSearchCV, 최적 모델 pkl 파일로 저장

    Args:
        X_train (_type_): Feature
        y_train (_type_): target

    """
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    xgb_model = XGBRegressor()

    parameters = {
        "max_depth": [1, 2, 3],
        "learning_rate": [0.05, 0.1, 0.3, 0.5],
        "n_estimators": [100, 200, 300, 400],
        "subsample": [0.1, 0.3, 0.5],
        "colsample_bytree": [0.1, 0.3, 0.5]
    }

    grid_xgb = GridSearchCV(
        xgb_model,
        param_grid = parameters,
        error_score='raise',
        n_jobs = -1,
        cv = 3
        )

    grid_xgb.fit(X_train, y_train)

    best_model = grid_xgb.best_estimator_

    return best_model

def model_eval(best_model, X_train, X_test, y_train, y_test):
    """평가 지표

    Args:
        best_model (_type_): 최적 파라미터를 가진 모델 객체
        X_train (_type_): 훈련 X
        X_test (_type_): 학습 X
        y_train (_type_): 훈련 y
        y_test (_type_): 학습 y
    """
    from sklearn.metrics import mean_squared_error, r2_score

    best_model.fit(X_train, y_train)

    y_pred_train = best_model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    y_pred_test = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    train_rmse = np.sqrt(abs(mse_train))
    test_rmse = np.sqrt(abs(mse_test))

    print("RMSE (train) : ", train_rmse)
    print("R2 (train) : ", r2_train)
    print("RMSE (test) : ", test_rmse)
    print("R2 (test) : ", r2_test)
