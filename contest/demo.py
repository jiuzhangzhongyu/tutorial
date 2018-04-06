import logging
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, MaxAbsScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import  pandas as pd

def lgbcv(X, y, X_test):

    estimator =  lgb.LGBMRegressor(objective='mse',n_estimators=100)
    pipe = Pipeline([
        ('scaler', MaxAbsScaler()),
        #('poly', PolynomialFeatures(interaction_only=True)),
        # ('select', SelectKBest(f_regression,k='all')),
        ('lgb', estimator)
        ])
    param_grid = [
    {
        # 'select__k':[10,20,26,32,'all'],
       'lgb__learning_rate':[0.05,0.1], 
       'lgb__num_leaves':[6,8,16,31],
       'lgb__subsample':[0.8,1.0],
    }
    ]

    gbm = GridSearchCV(pipe, param_grid, verbose=2, scoring='neg_mean_squared_error', cv=5)
    gbm.fit(X,y)
    
    logging.info(  "lgbcv {0}".format(gbm.best_score_ ))
    logging.info(  "lgbcv {0}".format(gbm.best_params_))

    score = gbm.predict(X_test)
