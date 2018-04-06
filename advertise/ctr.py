# coding: utf-8
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file

# import xgboost as xgb
import lightgbm as lgb
import logging,datetime,dateutil

import  numpy as np
from itertools import groupby
from operator import itemgetter
from feature import Feature,StatFeature,SuperStatFeature, FeatureType
import  argparse
"""
过滤条件:
raw_sample : user%10 == 0

behavior_log: user%10 == 0  & date >=20170506
"""


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='ctr.log',
                    filemode='a')

BIN_COLS  = ['adgroup_id' , 'pid',  'cate_id', 'campaign_id', 'customer','brand','cms_segid', 'cms_group_id']
VAL_COLS = [
        'final_gender_code','age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
        'price',  #adprice
    ]
def init_feature_list():
    logging.info("init feature list")
    buf = []
    for col in BIN_COLS:
        buf.append( Feature(name=col, prefix=col,startid=1,type= FeatureType.BIN, drop=False))
    for col in VAL_COLS:
        buf.append( Feature(name=col, prefix=col,startid=1,type= FeatureType.VAL, drop=False))

    return buf
def fill_feature_dict(fealist , df):
    logging.info('fill feature dict')
    map = {}
    for f in fealist:
        map[f.prefix] = f


    # cols = ['adgroup_id' , 'pid',  'cate_id', 'campaign_id', 'customer','brand','cms_segid', 'cms_group_id']
    for col in BIN_COLS:
        for v  in df[col].unique():
            fs = '{0}:{1}'.format(col, v)
            if col in map:
                fea = map[col]#type:Feature
                fea.tryAdd(col, fs, sep=':')


    for col in VAL_COLS:
        if col in map:
            fea = map[col]
            fea.tryAdd(col , col)

    start = 1
    for f in fealist:#type:Feature
        start = f.alignFeatureID(start)
        logging.info(f.coverRange())

    return start




def make_sample(fealist ,df, y ):
    logging.info('make sample')
    map = {}
    for f in fealist:
        map[f.prefix] = f


    cols = df.columns.values

    dvals = df.values

    r, c = df.shape

    k = 0
    for val in dvals:
        rbuf = []
        for i in range(0 , c):
            col = cols[i]
            v = val[i]

            fs = '{0}:{1}'.format(col, v)
            if col in map:
                fea = map[col]
                rbuf.append( fea.transform(col ,fs))

        rbuf.sort(key=lambda x: int(x.split(":")[0]))

        yield y[k] , ' '.join(rbuf)

        k +=1



def sample():
    ad_feature = pd.read_csv('ad_feature.csv',header=0)
    col='brand'

    # ad_feature[col] = ad_feature[col].astype('category', categories= ad_feature[col].unique()[~np.isnan(ad_feature[col].unique())])
    # print ad_feature['brand'].unique()
    # return
    user_profile = pd.read_csv('user_profile.csv',header=0)
    # raw_sample = pd.read_csv('raw_sample.csv',header=0)
    raw_sample = pd.read_csv('raw_sample.u0.sub.csv',header=0)

    print raw_sample.columns
    print user_profile.columns
    print ad_feature.columns
    # return
    # return
    raw_sample.drop(columns = ['nonclk'], inplace=True)

    ad_sample =  pd.merge(raw_sample, ad_feature, on='adgroup_id')
    user_ad_sample = pd.merge(ad_sample, user_profile, left_on='user', right_on='userid')

    y = user_ad_sample['clk']
    auxdf = user_ad_sample[['user','time_stamp']]

    user_ad_sample.drop(columns=['clk', 'user', 'userid','time_stamp'], inplace=True)

    fealist = init_feature_list()

    dimension = fill_feature_dict(fealist, user_ad_sample)

    with open('sample.txt', 'w') as f :
        i = 0
        auxvals = auxdf.values
        for label , feature  in make_sample(fealist, user_ad_sample, y):
            user ,timestamp = auxvals[i]
            day = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y%m%d')

            f.write('{0}:{1},{2},basic,{3},{4}\n'.format(user, day ,dimension , label , feature ))
            i+=1

    logging.info('sample done')

def init_beh_feature():
    fealist = [
        Feature(prefix='pv_cate', name='pv_cate', startid=1, type=FeatureType.BIN),
        Feature(prefix='cart_cate', name='cart_cate', startid=1, type=FeatureType.BIN),
        Feature(prefix='fav_cate', name='fav_cate', startid=1, type=FeatureType.BIN),
        Feature(prefix='buy_cate', name='buy_cate', startid=1, type=FeatureType.BIN),

        Feature(prefix='pv_brand', name='pv_brand', startid=1, type=FeatureType.BIN),
        Feature(prefix='cart_brand', name='cart_brand', startid=1, type=FeatureType.BIN),
        Feature(prefix='fav_brand', name='fav_brand', startid=1, type=FeatureType.BIN),
        Feature(prefix='buy_brand', name='buy_brand', startid=1, type=FeatureType.BIN),
    ]
    return fealist

def behavsample():
    """
    生成行为样本
    :return:
    """


    with open('behav.txt', 'w') as f:
        for key , dimension,type ,label,  units in make_behavsample():
            f.write('{0},{1},{2},{3},{4}\n'.format(key,dimension,type, label , units))

    logging.info('behavsample done')

def make_behavsample():
    """
    precondition: sort by date:user
    :return:
    """
    """
    ipv	浏览
cart	加入购物车
fav	喜欢
buy	购买
    """

    fealist = init_beh_feature()

    map = {}
    for f in fealist:
        map[f.prefix] = f


    def readline():

        with open('beh.u0.sorted.csv', 'r') as f:
            for L in f:
                user,time_stamp,btag,cate,brand = L.strip().split(",")
                #  add one day to align with ad click date.
                date = (datetime.datetime.fromtimestamp(int(time_stamp)) + datetime.timedelta(days=1)).strftime('%Y%m%d')
                yield '{0}:{1}'.format(user,date), (btag,cate, brand)

    #step 1:  fill dict .
    for key ,group in groupby(readline(), key=itemgetter(0)):
        user,date = key.split(':')
        for _, vals  in group:
            btag ,cate , brand = vals[0], vals[1], vals[2]
            feastr = '{0}_cate:{1}'.format(btag,cate)
            prefix = '{0}_cate'.format(btag)

            if prefix in map:
                fea =  map[prefix]
                fea.tryAdd(prefix=prefix, feastr=feastr)


            feastr = '{0}_brand:{1}'.format(btag,brand)

            prefix = '{0}_brand'.format(btag)

            if prefix in map:
                fea =  map[prefix]
                fea.tryAdd(prefix=prefix, feastr=feastr)
        pass

    start = 1
    for f in fealist:#type:Feature
        start = f.alignFeatureID(start)
        logging.info(f.coverRange())

    # step 2: transform featuer


    for key ,group in groupby(readline(), key=itemgetter(0)):
        user,date = key.split(':')
        rbuf = []
        for _, vals in group:
            btag, cate, brand = vals[0], vals[1], vals[2]
            for prefix , feastr in [
                ('{0}_cate'.format(btag),'{0}_cate:{1}'.format(btag,cate)  ),
                ('{0}_brand'.format(btag), '{0}_brand:{1}'.format(btag,brand))
            ]:
                if prefix in map:
                    fea =  map[prefix] #type:Feature
                    unit =  fea.transform(prefix=prefix, feastr=feastr)
                    if unit != -1:
                        rbuf.append(unit)
        rbuf.sort(key=lambda x: int(x.split(":")[0])) # make sure the dimension is ok .
        yield key , start , 'beh',  -1 , ' '.join(rbuf)
def merge_sample():
    with open('merge.s.txt', 'w') as f:
        for L in make_merge_sample():
            f.write(L+'\n')

    logging.info('merge_sample done')
def make_merge_sample(): #merge sample
    """
    precondition:  sort by key
    :return:
    """

    def readline():
        with open('merge.100w.sorted.csv','r') as f:
            for L in f:
                key , dimension, type , label, feature = L.strip().split(',')
                yield key , (dimension, type , int(label), feature)

    for key ,group in groupby(readline(),key=itemgetter(0)):
        map = {}
        real_label = -1
        # print group
        for _, vals in group:
            # print _
            dimension, typ , label , feature   = vals[0] ,vals[1],vals[2], vals[3]
            map[typ] = (int(dimension),feature)

            # print label, type(label)
            if int(label) != -1:
                real_label = label
                # print '--' * 8 , real_label , real_label != -1

        # print '++'* 8 , real_label ,real_label != -1
        if real_label != -1:
            d = 0
            rbuf = []
            fset = set()
            for typ in ['basic', 'beh']:
                if typ in map:
                    dimension , feature = map[typ]
                    for x in feature.split(' '):
                        fid, fval  = x.split(':')
                        fid = int(fid) + int(d)
                        if fid not in fset:
                            fset.add(fid)
                            rbuf.append('{0}:{1}'.format(fid, fval))
                d += dimension

            yield '{0} {1}'.format(real_label,  ' '.join(rbuf))




def behav():
    """
    非常大的数据，需要裁剪才能用于  python处理
    :return:
    """
    cnt = 0
    with open('beh.csv','w') as outf:
        with open('behavior_log.csv', 'r') as f:
            i = 0
            for L in f:
                user,time_stamp,btag,cate,brand =  L.split(",")
                if i == 0:
                    i+=1
                    continue
                # print time_stamp
                date = datetime.datetime.fromtimestamp(int(time_stamp)).strftime('%Y%m%d')
                if int(date) < 20170505 or (int(user))%10 >= 3:
                    cnt +=1
                else:
                    outf.write(L)
    logging.info('out the window , size = ' + str(cnt))

def main():
    """
    auc = 0.54
    :return:
    """
    X,  y =   load_svmlight_file('merge.s.txt', 1320000 )
    logging.info('load X, y done')

    cls = lgb.LGBMClassifier(objective='binary',num_leaves=31, n_estimators=100)

    param_grid = {
        'num_leaves': [16,31],
        'subsample': [0.8,1.0],
    }
    gs = GridSearchCV(estimator=cls, param_grid=param_grid, cv= 3 , scoring='roc_auc', verbose=True)
    gs.fit(X, y)

    logging.info('best score = {0}'.format(gs.best_score_))
    logging.info('best params = {0}'.format(gs.best_params_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=False,default='merge')
    arg = parser.parse_args()
    if arg.mode == 'sample':
        sample()
    elif arg.mode == 'behav':
        behavsample()
    elif arg.mode == 'merge':
        merge_sample()
    elif arg.mode == 'train':
        main()


