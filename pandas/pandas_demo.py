#encoding=utf-8
__author__ = 'light'
import pandas as pd
import numpy as np

from datetime import  datetime
from dateutil import  relativedelta
"""
pandas demo for jiuzhang seminar.
Prerequriement:
pip install numpy pandas matplotlib
pip install xlrd
pip install sqlalchemy
"""

def miscellaneous():
    """

    :return:
    """

    # Text
    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])

    # str.func  func 就是python string 能用的一些 function
    print df1['A'].str.lower()
    print df1['A'].str[0]

    # plot
    import  math
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'x': range(10) , 'y' : [math.sin(i) for i in range(10)] })
    df.plot('x','y')
    plt.show()


    df.plot('x','y',kind='scatter')

    plt.show()

    ###Option
    """
    Option :  http://pandas.pydata.org/pandas-docs/stable/options.html#
    """

    df = pd.DataFrame(np.random.randn(5,5))

    pd.set_option('precision',7)

    print df

    pd.set_option('precision', 4)
    print df









    pass
def datetime_demo():
    """
    to_datetime   date_range  DateOffset ?
    :return:
    """
    day = pd.to_datetime('20170123')

    df1 = pd.DataFrame({'day': ['20170101','20170102'] , 'demand': [ 100 , 120] })

    df1['day'] = df1['day'].apply(lambda  x : pd.to_datetime(x))
    # df1['day'] = pd.to_datetime(df1['day'])  is also right.


    # date_range
    pd.date_range(start = '20170101' ,end = '20170201' )
    pd.date_range(start = '20170101' ,periods=10)
    pd.date_range(start = '20170101' ,periods=10, freq='W')  # ten week
    pd.date_range(start = '20170101' ,periods=10, freq='M')  # ten monthes
    pd.date_range(start = '20170101' ,periods=10, freq='MS')  # ten monthes
    pd.date_range(start = '20170102' ,periods=10, freq='MS')  # ten monthes  MS: month start.
    """

    所有的 freq string 会被转成  DateOffset subclass,之后再执行
    Alias	Description
    B	business day frequency
    C	custom business day frequency
    D	calendar day frequency
    W	weekly frequency
    M	month end frequency
    SM	semi-month end frequency (15th and end of month)
    BM	business month end frequency
    CBM	custom business month end frequency
    MS	month start frequency
    SMS	semi-month start frequency (1st and 15th)
    BMS	business month start frequency
    CBMS	custom business month start frequency
    Q	quarter end frequency
    BQ	business quarter end frequency
    QS	quarter start frequency
    BQS	business quarter start frequency
    A, Y	year end frequency
    BA, BY	business year end frequency
    AS, YS	year start frequency
    BAS, BYS	business year start frequency
    BH	business hour frequency
    H	hourly frequency
    T, min	minutely frequency
    S	secondly frequency
    L, ms	milliseconds
    U, us	microseconds
    N	nanoseconds
    """


    # DateOffset

    d = datetime(2008, 8, 18, 9, 0)
    d + relativedelta(months=4, days=5)


    from pandas.tseries.offsets import *


    d + DateOffset(months=4, days=5)


    offset = BMonthEnd()
    offset.rollback(d)
    offset.rollforward(d)









    pass
def merge_join_concatenate():

    ##concatenate
    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
    df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                         'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                       'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])

    pd.concat([df1,df2],axis=0)

    pd.concat([df1,df2],axis=1)  #注意： concat 时 按照 index对齐的


    #默认 index 从 0 开始编码
    df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                         'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                       'D': ['D4', 'D5', 'D6', 'D7']},
                       )

    pd.concat([df1,df2],axis=1)


    df = pd.concat([df1,df2],axis=0)
    df.reset_index().drop(columns =['index'])


    ## merge and join
    """
    pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
    """

    left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                    'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})

    right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                         'D': ['D0', 'D1', 'D2', 'D3']})

    result = pd.merge(left, right, on='key')


    right = pd.DataFrame({'right_key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                         'D': ['D0', 'D1', 'D2', 'D3']})

    result = pd.merge(left, right ,left_on='key', right_on= 'right_key', how = 'inner')


    ## 按照 index  join
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                        'B': ['B0', 'B1', 'B2']},
                            index=['K0', 'K1', 'K2'])


    right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                              'D': ['D0', 'D2', 'D3']},
                              index=['K0', 'K2', 'K3'])


    result = left.join(right)

    pd.merge(left, right , how='inner', left_index=True, right_index=True)





def groupby():
    df = pd.read_csv('data/ad_feature.csv',header=0) #type:pd.DataFrame
    df.groupby('cate_id').count()
    df.groupby('cate_id')['price'].mean()
    df.groupby('cate_id')['price'].mean().reset_index()
    df.groupby('cate_id')['price'].max()
def missing_value_and_outlier():
    """

    :return:
    """

    df = pd.read_csv('data/ad_feature.csv',header=0) #type:pd.DataFrame

    """
    adgroup_id          0
    cate_id             0
    campaign_id         0
    customer            0
    brand          246330
    price               0
    dtype: int64
    """

    df.isna().sum()
    # outlier clip

    des =  df['price'].describe()

    valid_max = des['75%'] + 3 * ( des['75%'] - des['50%'])

    df['price'] = df['price'].clip(0 , valid_max)

    #重新查看price statistics.
    df['price'].describe()

    #########missing_value  fillna

    data2 = [{'a': 1, 'b': 2}, {'a': 5,  'c': 20}]
    print df.isna().sum()

    df= pd.DataFrame(data2)
    df.fillna(df.median())
    df.fillna(0.)




def index_selecting():
    """
    index and selecting data.
    :return:
    """
    df = pd.read_csv('data/ad_feature.csv',header=0) #type:pd.DataFrame
    print df[:2] , df[2: ] #前两行

    df.iloc[:2 , :]
    df.iloc[:2,  [2,3] ]  # 第 2 列和 第3列

    #  df.loc[row_indexer,column_indexer]
    df.loc[3, ['cate_id','price']]

    df[['cate_id', 'price']]

    #boolean index
    df[ df['price'] > 1000]
    df[ (df['price'] > 1000) & (df['price'] < 2000)]



    df[df['cate_id'].isin([6261])]

    #select by callable


    # .loc, .iloc, and also [] indexing can accept a callable as indexer


    df.loc[lambda d: d.price > 2000, :]


def io():
    """
    input and output
    :return:
    """

    df = pd.read_csv('data/ad_feature.csv',header=0)
    print df.columns
    print df.shape
    df.to_csv('output/ad_feature.csv',index=False, header=True)
    df[:5].to_html('output/ad.html', header=True)

    from sqlalchemy import create_engine

    engine = create_engine('sqlite:///:memory:')
    df[:5].to_sql('adfeature', engine)
    sqldf = pd.read_sql_query('select * from adfeature' , engine)




    df = pd.read_excel('data/train_ai_factory.xlsx', header=0)
    print df.columns , df.shape

def data_structure():
    """
    Series/DataFrame/Panel
    Panel is less useful in practice.

    todo:  use an array to sub np.random.
    :return:
    """

    items = [1.0, 2.0, 3.0, 4.0, 5.0 ]
    s = pd.Series(items, index=['a', 'b', 'c', 'd', 'e'])
    # s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    print s
    s = pd.Series(items)
    print s

    d= {'one': [1.0, 2.0, 3.0, 4.0], 'two': [4.0, 3.0, 2.0, 1.0]}

    df = pd.DataFrame(d)
    print df
    df = pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
    print df


    data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
    df = pd.DataFrame(data2)

    print df

def essential_basic_function():
    """
    Head/Tail
    Statistics
    Iteration
    Sort
    Dtype (astype)
    Tablewise/Row or Column-Wise/Transform/Elementwise
    Reindex /Reset_index

    :return:
    """
    df = pd.read_csv('data/ad_feature.csv',header=0) #type:pd.DataFrame

    print df.head(10)  #default 5
    print df.tail(10)  #default 5

    print df['price'].describe()
    print df['price'].mean(),  df['price'].std() , df['price'].max() , df['price'].min(), df['price'].median() , df['price'].mad() ,df['price'].skew() , df['price'].kurt()


    # iteration
    """
    DataFrame.iteritems()   =>  Iterator over (column name, Series) pairs.
    DataFrame.iterrows()    =>  Iterate over DataFrame rows as (index, Series) pairs.
    DataFrame.itertuples(index=True, name='Pandas')[source] =>  Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple.
    """
    for row in df[:5].itertuples():
        print row.cate_id, row.price
    for row in df[:5].iterrows():
        print row['cate_id'], row['price']

    for col , ser  in df[:2].iteritems():
        print col, ser

    # sort  sort_values  and sort_index()
    # rdf =  df[:2].reindex([1,0])  #改变顺序.
    #
    # rdf.sort_index()


    pricedf =  df[:10].sort_values(by='price')

    pricedf.sort_index()


    # reindex  reset_index()

    df[:2].reindex([1,0])  # baipeng

    df[:2].reindex([1,0]).reset_index()  # baipeng


    # type: dtypes
    """
    adgroup_id       int64
    cate_id          int64
    campaign_id      int64
    customer         int64
    brand          float64
    price          float64
    dtype: object

    """
    print df.dtypes

    print df['cate_id'].astype('float64') #转换float

    #### Table-wise
    df[:2].pipe(lambda x : x[['cate_id','price']] )

    #### Row or Column Wise
    df['price'].apply(lambda x : min(x, 100))
    #### Elementwise
    df[:2].applymap(lambda x : min(x,1000))

def main():
    io()
    raw_input('\t\tPress any key to continue')
    data_structure()
    raw_input('\t\tPress any key to continue')
    essential_basic_function()
    raw_input('\t\tPress any key to continue')
    index_selecting()
    raw_input('\t\tPress any key to continue')
    missing_value_and_outlier()
    raw_input('\t\tPress any key to continue')
    groupby()
    raw_input('\t\tPress any key to continue')
    merge_join_concatenate()
    raw_input('\t\tPress any key to continue')
    datetime_demo()
    raw_input('\t\tPress any key to continue')
    miscellaneous()

if __name__ == '__main__':
    main()
