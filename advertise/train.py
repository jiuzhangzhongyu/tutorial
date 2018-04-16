# encoding=utf-8
"""
@author : peng
"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import fire
import logging
import pandas as pd
import  datetime
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',
                    filemode='a')



def lgbcv(sample=None):
    """
     with empty feature:  0.63381992881
     with full feature:  0.738394062041   ->太多的空特征 对auc有很大影响
    :return:
    """

    print sample
    X, y, qids  = load_svmlight_file(sample,query_id=True)

    param_grid = [{
        'lgb__learning_rate': [0.1],
        'lgb__num_leaves': [31 ],
        'lgb__subsample': [ 1.0],
    }]


    estimator = lgb.LGBMClassifier(objective="binary", n_estimators=100)

    pipe = Pipeline([('lgb', estimator)])
    scoring = {
        'AUC': 'roc_auc',
        'logloss': 'neg_log_loss',
        'ACCURACY': 'accuracy',
    }
    gbm = GridSearchCV(pipe, param_grid=param_grid, verbose=2, scoring=scoring, cv=5, refit='logloss') # neg_log_loss
    gbm.fit(X, y)


    logging.info(sample)
    logging.info(gbm.cv_results_)
    logging.info("{0} lgbcv {1}".format(sample, gbm.best_score_))
    logging.info("lgbcv {0}".format(gbm.best_params_))


if __name__ == '__main__':
    fire.Fire()
