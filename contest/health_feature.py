# encoding=utf-8
"""
@author : ZhongQing
"""
from feature import Feature ,FeatureType

import logging
import  pandas as pd
import numpy as np

Numerical = 'Numerical'
Categorical = 'Categorical'
Mix = 'Mix'


def init_feature_list():
    logging.info("init_feature_list")


    df = pd.read_csv('colmap.csv',header=0)
    df['table_id'] = df['table_id'].astype(str)

    buf = []





    for row in df.itertuples():
        if row.type == Numerical:
            ft = FeatureType.VAL
        elif row.type == Categorical:
            ft = FeatureType.BIN
        elif row.type == Mix:
            ft = FeatureType.MIX
        else:
            raise NotImplementedError("Check your TYPE")

        f = Feature(name=row.table_id, prefix=row.table_id, startid=1, type=ft)
        buf.append(f)


    return  buf

def fill_feature_dict(fealist , X):
    logging.info('fill feature dict')
    map = {}

    bincols = []
    for f in fealist:
        map[f.prefix] = f
        if f.type == FeatureType.BIN or f.type == FeatureType.MIX:
            bincols.append(f.prefix)


    for row in X.itertuples():
        feature_str = row.X
        us = feature_str.split(" ")

        for unit in us:
            try:
                k, v  = unit.split("=",1)
                if k in bincols:
                    f = map[k]  # type:Feature
                    f.tryAdd(k, unit)
            except:
                print( 'ERROR=', unit)


    start = 1

    for f in fealist:  # type:Feature
        start = f.alignFeatureID(start)
        logging.info(f.coverRange())

    return start
def transform_feature(fealist, X):
    map = {}

    bincols = []
    for f in fealist:
        map[f.prefix] = f
        if f.type == FeatureType.BIN:
            bincols.append(f.prefix)



    buf = []
    for row in X.itertuples():
        lbuf = []
        feature_str = row.X
        us = feature_str.split(" ")

        fidset = set()
        for unit in us:
            try:
                k, v = unit.split("=", 1 )
                if k in map:
                    f = map[k] #type:Feature
                    sparse_f = f.transform(k , unit)
                    if sparse_f != -1:
                        fid , _  = sparse_f.split(':')

                        if sparse_f != -1 and fid not in fidset:
                            lbuf.append(sparse_f)
                            fidset.add(fid)
            except Exception as e:
                logging.warn(e)
                logging.warn(unit)

        lbuf.sort(key=lambda x: int(x.split(":")[0]))

        buf.append(' '.join(lbuf))

    X['sparse'] = buf

    X[['vid','sparse']].to_csv('X_libsvm.csv',index=False)


