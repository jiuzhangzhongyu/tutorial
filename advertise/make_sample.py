# encoding=utf-8
"""
@author : peng
"""
from feature import Feature,FeatureType
import logging
import pandas as pd
BIN_COLS = ['item_id', 'item_brand_id', 'shop_id']
VAL_COLS = ['item_posrate', 'recent_15minutes', 'shop_score_delivery']


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
            fs = '{0}={1}'.format(col, v)
            if col in map:
                fea = map[col] #type:Feature
                fea.tryAdd(col, fs)

    start = 1
    for f in fealist:  # type:Feature
        start = f.alignFeatureID(start)
        logging.info(f.coverRange())

    return start


def make_sample(fealist ,df, y, qidvals ):
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

            fs = '{0}={1}'.format(col, v)
            if col in map:
                fea = map[col]
                rbuf.append( fea.transform(col ,fs))

        rbuf.sort(key=lambda x: int(x.split(":")[0]))

        yield y[k] , qidvals[k], ' '.join(rbuf)

        k +=1

def main():
    df = pd.read_csv('./conv_ins.csv',sep=' ')
    Y = df['is_trade']
    qidvals = df['instance_id']
    fealist = init_feature_list()
    fill_feature_dict(fealist, df)

    with open('final.sample', 'w') as f :

        for label , qid, feature  in make_sample(fealist, df, Y, qidvals):
            f.write('{} qid:{} {}\n'.format(label,qid, feature))

    logging.info('sample done')

if __name__ == '__main__':
    main()


