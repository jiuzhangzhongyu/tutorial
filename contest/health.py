# encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


"""
@author : ZhongQing


病人id、体检项目id 和体检结果

每个人 体检项目不一样；有的多，有的少.

体检项目 part1 有 317个； part2 有  2483 个；

有的体检项目做的人多，有的体检项目做的人少 . 特征多少的问题.

可能是要找出各种体检项目的关联性；

最简单套路就是 分类再回归等思路.

中文处理:

每个条目识别关键字 进行分裂么?肯定是要结构化，之后再用pivot_table就方便了.

从数据看，还是得吧描述解开，然后再进行训练.

"""



import fire, logging, csv
import  pandas as pd
import time ,datetime
import numpy as np
import jieba
from sklearn.datasets import load_svmlight_file
import lightgbm as lgb
import re
import pickle
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

jieba.add_word('未发现明显异常')
jieba.add_word('未发现异常')
jieba.add_word('未见异常') ###############big
jieba.add_word('未见明显异常')
jieba.add_word('未检出明显异常')
jieba.add_word('未检出异常')

jieba.add_word('阴性')
jieba.add_word('阳性')
# jieba.add_word('未见')
jieba.add_word('未检出')
jieba.add_word('正常')

from health_feature import  init_feature_list, fill_feature_dict, transform_feature

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='health.log',
                    filemode='a')

def check_contain_chinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

Glossary = {}
def _make_vid_X(df):
    """
    所有的都转成 string以后，再处理.
    :param df: 单个vid的 df的值 ;包含了这个人所有的体检记录.
    :return:
    """
    buf = []



    for row in df.itertuples():
        # row.vid  row.table_id  row.field_results  有的存在多个值.
        try:
            v = float(row.field_results)
            str = '{}={}'.format(row.table_id, v)
            buf.append(str)
        except:
            # if row.field_results

            # print row.field_results
            """可能含有中文字符"""
            temp = row.field_results
            if check_contain_chinese(temp):
                temp =  re.sub('\s+', '_' , temp)
                if len(temp) <= 4*3 or (temp.startswith('未') and len(temp) < 8* 3):
                # 10个字以下，很多是专有名词，不进行切词.
                    buf.append('{}={}'.format(row.table_id, temp))
                    if temp not in Glossary:
                        Glossary[temp] = 1
                        logging.info('glossary {}'.format(temp))
                    else:
                        Glossary[temp] = Glossary[temp] + 1
                else:

                    temp = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), temp)
                    words = jieba.lcut(temp)
                    for w in words:
                        # print w
                        str = u'{}={}'.format(row.table_id, w)
                        buf.append(str)
            else:
                t =  re.sub('\s+', '_' , temp)

                # raw_input("\t\tPress any")

                str = '{}={}'.format(row.table_id, t)
                buf.append(str)

    return ' '.join(buf)



def _conv_field(x):
    """
    转化基本的field
    :param x:
    :return:
    """
    x = str(x)
    x = re.sub('[><]', '', str(x))

    NEG  = '阴性'
    POS  = '阳性'
    Normal = '正常'
    convmap = {
        '-' : NEG,
        '+' : POS,
    }
    if x in convmap:
        x = convmap[x]
    else:
        if x.startswith(NEG):
            x = NEG
        elif x.startswith(POS):
            x = POS

        if  x == 'Normal':
            x = Normal
    return x



def identify_columns():
    """
    识别列是 Numerical 还是别的.

    :return:
    """

    part1 = 'data/meinian_round1_data_part1_20180408.txt'
    part2 = 'data/meinian_round1_data_part2_20180408.txt'

    df1 = pd.read_csv(part1, sep='$', encoding='utf-8')
    df2 = pd.read_csv(part2, sep='$', encoding='utf-8')


    df = pd.concat([df1,df2], axis=0).reset_index(drop=True)

    df['table_id'] = df['table_id'].astype(str)

    # df['field_results'] = df['field_results'].apply(lambda x:  re.sub('[><]', '' , str(x) ))
    df['field_results'] = df['field_results'].apply(_conv_field)

    # df['field_result'] = df['field_results'].map({'-': '阴性'})
    colmap = {}
    Numerical = 'Numerical'
    Categorical = 'Categorical'
    Mix = 'Mix'

    for row in df.itertuples():
        try:
            ##有大量NaN float(nan)也能得到结果
            v = float(row.field_results)

            # if v > 0:
            #     pass
            # else:
            #     print v, row.field_results
            #     raw_input('\t\tPress any')

            if str(row.table_id) not in colmap:
                colmap[str(row.table_id)] = Numerical
            else:
                if colmap[row.table_id] != Numerical:
                    colmap[row.table_id] = Mix
                else:
                    pass #keep the Numerical



        except:
            if str(row.table_id) not in colmap:
                colmap[str(row.table_id)] = Categorical
            elif row.table_id in colmap:
                if colmap[row.table_id] != Categorical:
                    colmap[row.table_id] = Mix



    buf = []
    c = []
    for k  in colmap:
        buf.append(k)
        c.append(colmap[k])

    df = pd.DataFrame({'table_id': buf, 'type': c })
    df.to_csv('colmap.csv',index=False)


    print( df.groupby('type').count())


    """
     df.groupby('type').count()
Out[23]:
             table_id
type
Categorical      1460 =>非常多的.
Numerical        1335

    """

def make_X():

    """
    生成基础的X
    :return:
    """

    part1 = 'data/meinian_round1_data_part1_20180408.txt'
    part2 = 'data/meinian_round1_data_part2_20180408.txt'
    df1 = pd.read_csv(part1, sep='$', encoding='utf-8', chunksize=100, iterator=True).next()
    df2 = pd.read_csv(part2, sep='$', encoding='utf-8', chunksize=100, iterator=True).next()


    df1 = pd.read_csv(part1, sep='$', encoding='utf-8')
    df2 = pd.read_csv(part2, sep='$', encoding='utf-8')


    df = pd.concat([df1,df2], axis=0).reset_index(drop=True)

    df['field_results'] = df['field_results'].apply(_conv_field)

    dfx = df.groupby('vid').apply(_make_vid_X)

    dfx.to_csv('X.csv',sep=' ')

def make_libsvm_X():
    """
    先定义 feature,fillFeature ;
    之后再进行transform>
    :return:
    """
    X = pd.read_csv('X.csv', sep=' ', header=None)
    X.columns = ['vid', 'X']
    fealist = init_feature_list()
    fill_feature_dict(fealist, X )
    transform_feature(fealist, X)


def _local_metric(X, y ,col):
    """
    经过几番实验，血清甘油三酯 采用log ；  mse log是最低的. 后续可以实验.
    :param X:
    :param y:
    :param col:
    :return:
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    logp =  True
    if col == '血清甘油三酯' and logp:
        y_train = np.log1p(y_train)
        # y_train = y_train/2
        # y_train = (y_train - 1.6)/1.3

    y_train = pd.Series(y_train).clip(0)
    y_test = pd.Series(y_test).clip(0)


    reg = lgb.LGBMRegressor(objective='mse', num_leaves=31, n_estimators=100)
    reg.fit(X_train, y_train)
    
   # rf = lgb.LGBMRegressor(boosting_type = 'dart', n_estimators=100,  bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8)
    #rf.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    #y_rf = rf.predict(X_test)
    
    if col == '血清甘油三酯' and logp:
        y_pred = np.e ** y_pred -1
#        y_rf = np.e ** y_rf -1 
        # y_pred = y_pred * 2
        # y_pred = y_pred * 1.3 +1.6

    logmean = mean_squared_error( np.log1p(y_test) , np.log1p(y_pred))
    logging.info('local metric {} = {} '.format(col, logmean))



def make_estimator(df, col):
    """

    :param df: vid sparse
    :param col:
    :return:
    """
    logging.info("make_estimator {}".format(col))
    libsvm = '{}.libsvm'.format(col)

    print ( df[[col, 'sparse']].head())
    print (df[col].describe())
    #df[col] = df[col].astype(float, errors = 'ignore')
    if df.dtypes[col] == float:
        pass
    else:

        pattern = r'[0-9]+'
        df = df[df[col].str.contains(pattern)]
        df[col] = df[col].str.extract(r'([0-9]+[.]?[0-9]*)',expand=False)
        print( df[col].describe())

    df[[col, 'sparse']].to_csv(libsvm, sep=' ', header=False,index=False)
    subprocess.check_call('sed -i s/\\"//g {}'.format(libsvm),shell=True)
    # subprocess.check_call('sed -i s/14406:未见异常/14406:2/g {}'.format(libsvm),shell=True)
    X, y = load_svmlight_file(libsvm, n_features=220000)

    if col == '血清甘油三酯':
        print ('train 甘油三酯')
        y_solid = np.log1p(y)
    else:
        y_solid = y
    reg = lgb.LGBMRegressor(objective='mse', num_leaves=31, n_estimators=100)
    reg.fit(X, y_solid)


    _local_metric(X,y , col)
    return reg

def conv(x):
    try:
        x = float(x)
    except:
        us = re.findall(r'[0-9]+[.]?[0-9]*', x)
        if len(us) > 0 :
            x = float(us[0])
        else:
            x = -1
    return x



def fit_and_test():
    logging.info("fit and test")

    train_path = 'data/meinian_round1_train_20180408.csv'
    train = pd.read_csv(train_path, encoding='gbk')
    t = train.iloc[:,1:].applymap(conv)
    t = t.clip(-1, t.median() * 10, axis=1)
    t['vid'] =train['vid']
    train = t
    """
        训练集中vid总共有 57298 人； 测试集a中大概有 9538人
    """
    # map = {}
    # for  col  in [u'收缩压', u'舒张压',u'血清甘油三酯',u'血清高密度脂蛋白',u'血清低密度脂蛋白']:
    #     cls = make_cls(col, df1, df2, train_target)
    #
    #     map[col] = cls
    #
    #
    # #仅仅包含了 人的姓名. 需要把对应特征构造出来: 这些数据没有时间概念 .
    #test_path = 'data/meinian_round1_test_a_20180409.csv'
    test_path = 'data/meinian_round1_test_b_20180505.csv'
    test = pd.read_csv(test_path, encoding='gbk')
    #

    X = pd.read_csv('X_libsvm.csv',header=0)
    # X.columns = ['vid', 'X']

    df = pd.merge(train , X,  on = 'vid'  ) #没有丢失信息.


    map = {}
    for  col  in [u'收缩压', u'舒张压',u'血清甘油三酯',u'血清高密度脂蛋白',u'血清低密度脂蛋白']:
    # for  col  in [u'血清甘油三酯']:
        es = make_estimator(df, col)
        map[col] = es

    logging.info("test shape = {}".format(test.shape))
    test = pd.merge(test, X, on='vid',how='left')
    logging.info("after merge, test shape = {}".format(test.shape))
    
    test['y'] = 0

    test[['y','sparse']].to_csv('test.libsvm',sep=' ',header=False,index=False)

    subprocess.check_call('sed -i s/\\"//g test.libsvm', shell=True)
    
    X_test, _ = load_svmlight_file('test.libsvm',n_features=220000)
    for col in [u'收缩压', u'舒张压', u'血清甘油三酯', u'血清高密度脂蛋白', u'血清低密度脂蛋白']:
        if col in map:
            es = map[col]
            score = es.predict(X_test)
            if col == u'血清甘油三酯':
                score = np.e ** score -1
            test[col] = [ "%.3f" % i for i in  score]

    day = datetime.datetime.today().strftime("%Y%m%d")
    test[['vid', u'收缩压', u'舒张压', u'血清甘油三酯', u'血清高密度脂蛋白', u'血清低密度脂蛋白']].to_csv('submit/{}.health.submit.g.csv'.format(day),header=False,index=False)

    logging.info("done")



def flow():

    logging.info("flow begin")
    identify_columns()
    make_libsvm_X()
    fit_and_test()
    logging.info("flow end")

if __name__ == '__main__':

    fire.Fire()
