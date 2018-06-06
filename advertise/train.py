# encoding=utf-8
"""
@author : peng
"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import fire
import logging
import pandas as pd
import scipy.sparse as sp
import numpy as np
import  datetime
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='train.log',
                    filemode='a')

def lgbtrain(debug = False):
    """
    baseline   about:
    use lightgbm categorical feature:  0.0906228
    use onehot :
    {'valid_0': {'binary_logloss': 0.08957243049867385}}) , auc = 0.6764937293294

    use posrate and instance feature:
    0.0887057851966679 auc = 0.69

    :param debug:
    :return:
    """
    path = 'data/round1_ijcai_18_train_20180301.txt'
    path = 'data/conv_ins.csv'
    if debug:
        nrows = 100000
    else:
        nrows = 100 * 10000 * 10
    df = pd.read_csv(path ,sep=' ' ,header=0, nrows = nrows)

    y = df.pop('is_trade')

    poplist = ['instance_id','item_property_list',
               'user_id', 'context_timestamp', 'context_id',
               'predict_category_property',
               'date'
               ]

    enc = LabelEncoder()
    categorical_list = ['item_id', 'item_brand_id', 'item_city_id','user_gender_id', 'user_occupation_id', 'shop_id' ]

    continous_list = [u'item_price_level', u'item_sales_level', u'item_collected_level',
                      u'item_pv_level', u'user_age_level',u'user_star_level',
                      u'shop_review_num_level',u'shop_review_positive_rate',
                      u'shop_star_level',u'shop_score_service', u'shop_score_delivery',
                      u'shop_score_description'
                      ]

    continous_list = [u'item_price_level', u'item_sales_level', u'item_collected_level',
                      u'item_pv_level', u'user_age_level', u'user_star_level',
                      u'shop_review_num_level', u'shop_review_positive_rate',
                      u'shop_star_level', u'shop_score_service', u'shop_score_delivery',
                      u'shop_score_description',

                      #####下面这些是我们自己构造的.
                      u'item_posrate', u'age_item_posrate' , u'first_to_now', u'prev_to_now',  u'recent_15minutes',
                      ]
    vector_list = ['item_category_list']
    for col in poplist:
        if col in df.columns:
            df.pop(col)

    space = None

    feature_buf  = []
    for col in continous_list:
        if col in df.columns:
            colval = df[col].values.reshape(-1, 1 )
            if space is None:
                space  = colval
            else:
                space = np.hstack((space, colval))
            feature_buf.append(col)
            print ( "continuous shape  = " , space.shape, "type  = " , type(space))


    #有些负数存在.
    for cc in categorical_list:
        df[cc] = enc.fit_transform(df[cc])
        oneenc = OneHotEncoder(sparse=True)
        colval = oneenc.fit_transform(np.reshape( df[cc] , (-1, 1)))
        print ("{} shape = {}" .format(cc , colval.shape))
        space = sp.hstack((space, colval))
        print ("{} hstack ok ".format(cc))

        for i in range(colval.shape[1]):
            feature_buf.append(cc)

    for cc in vector_list:
        cv= CountVectorizer()
        colval  = cv.fit_transform(df[cc])
        print ("{} shape = {}".format(cc, colval.shape))
        space = sp.hstack((space, colval))
        print ("{} ok".format(cc))

        for i in range(colval.shape[1]):
            feature_buf.append(cc)

    logging.info('Xshape = {}'.format(space.shape))

    X = space
    X_train ,X_test, y_train ,y_test = train_test_split(X, y , test_size=0.3, random_state=0)
    estimator = lgb.LGBMClassifier(objective="binary", n_estimators=10000, learning_rate=0.05, subsample=0.8 , subsample_freq=1)
    # estimator.fit(X_train, y_train, categorical_feature=categorical_list, eval_set=[(X_test, y_test)], eval_metric=['binary_logloss'] , early_stopping_rounds=20)
    estimator.fit(X_train, y_train,  eval_set=[(X_test, y_test)], eval_metric=['binary_logloss'] , early_stopping_rounds=20)

    auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:,1])
    logging.info('debug = {} best iteration = {} , best_score = {} , auc = {}'.format(debug,  estimator.best_iteration_ , estimator.best_score_ , auc))

    pd.DataFrame( {'feature': feature_buf ,  'importance': estimator.feature_importances_ }).to_csv('imp.csv', index=False)


def lgbcv(sample=None):
    """
     with empty feature:  0.63381992881
     with full feature:  0.738394062041   ->太多的空特征 对auc有很大影响
    :return:
    """

    print (sample)
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
