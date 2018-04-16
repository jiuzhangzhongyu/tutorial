#encoding=utf-8
__author__ = 'peng'



import fire, logging
import  pandas as pd
import time ,datetime
import numpy as np
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='ad.log',
                    filemode='a')

class Predict_Category_Property(object):
    def __init__(self, line):
        units  = line.split(';')

        buf = []
        for u in units:
            try:
                cate, ps = u.split(':')
                pss = ps.split(',')

                for p in pss:
                    buf.append( (cate , p ))
            except:
                logging.warn(line)

        self.buf = buf

    def __iter__(self):

        for cate , property in self.buf:
            yield cate, property


def convdate(t):

    return time.strftime('%Y-%m-%d', time.localtime(t))

def convhour(t):
    return time.strftime('%H', time.localtime(t))

def conv_item_property_list(key, cl):
    us = cl.split(";")
    buf = []
    for u in us:
        buf.append('{}={}'.format(key, u))
    buf.append('item_property_len={}'.format(len(us)))
    return buf
def conv_category_list(cl):
    us = cl.split(";")
    buf = []
    for u in us:
        buf.append(u)
    return buf

def conv_property_list(cl):
    us = cl.split(";")
    buf = []
    for u in us:
        buf.append(u)
    return buf
def add_category(metabuf, row):
    for c in conv_category_list(row.item_category_list):
        metabuf.append('c_user_category={uid}_{category}'.format(uid=row.user_id, category=c))

def add_property(metabuf, row):
    for c in conv_property_list(row.item_property_list):
        metabuf.append('c_user_property={uid}_{property}'.format(uid=row.user_id, property=c))


def conv_item_category_list(key, hour,  cl):

    us = cl.split(";")
    buf = []
    for u in us:
        buf.append('{}={}'.format(key, u))
        buf.append('{}#context_hour={}#{}'.format(key, u,hour))
    return buf
def add_predict_category_property(metabuf, row):
    line = row.predict_category_property
    pcp = Predict_Category_Property(line)

    pc = []
    for cate, prop in pcp:
        pc.append(cate)
        metabuf.append('c_user_pprop={uid}_{pprop}'.format(uid=row.user_id, pprop=prop))
    pc = set(pc)
    for c in pc:
        metabuf.append('c_user_pcate={uid}_{pcate}'.format(uid=row.user_id, pcate=c))

def conv_predict_category_property(line):
    pcp = Predict_Category_Property(line)
    buf = []

    pc = []
    for cate, prop in pcp:
        pc.append(cate)
        buf.append('predict_prop={}'.format(prop))
    pc = set(pc)
    for c in pc:
        buf.append('predict_cate={}'.format(c))


    return buf
def search_category_corr(row):
    item_category_set = conv_category_list(row.item_category_list)
    pcp =  Predict_Category_Property(row.predict_category_property)


    hit = 0
    for cate ,prop in pcp:
        for idx, item_category in enumerate(item_category_set):
            if cate == item_category:
                hit = idx +1


    return hit
    # pass
def search_property_corr(row):

    item_property_set = conv_property_list(row.item_property_list)
    pcp =  Predict_Category_Property(row.predict_category_property)

    hit = 0
    for cate ,prop in pcp:
        for idx, item_property in enumerate(item_property_set):
            if prop  == item_property:
                hit += 1
    return hit

def augment_bin(row, buf, hour):

    buf.append('bin_item_price_level={}'.format(row.item_price_level))
    buf.append('bin_item_sales_level={}'.format(row.item_sales_level))
    buf.append('bin_item_collected_level={}'.format(row.item_collected_level))
    buf.append('bin_item_pv_level={}'.format(row.item_pv_level))
    buf.append('bin_context_hour={}'.format(int(hour)))
    buf.append('bin_user_star_level={}'.format(row.user_star_level))
    buf.append('bin_user_age_level={}'.format(row.user_age_level))
    buf.append(u'bin_shop_review_num_level={}'.format(row.shop_review_num_level))
    buf.append(u'bin_shop_star_level={}'.format(row.shop_star_level))

def main():
    """
    Index([u'instance_id', u'item_id', u'item_category_list',
       u'item_property_list', u'item_brand_id', u'item_city_id',
       u'item_price_level', u'item_sales_level', u'item_collected_level',
       u'item_pv_level', u'user_id', u'user_gender_id', u'user_age_level',
       u'user_occupation_id', u'user_star_level', u'context_id',
       u'context_timestamp', u'context_page_id', u'predict_category_property',
       u'shop_id', u'shop_review_num_level', u'shop_review_positive_rate',
       u'shop_star_level', u'shop_score_service', u'shop_score_delivery',
       u'shop_score_description', u'is_trade'],
      dtype='object')



    数据拼接格式为 "category_0;category_1;category_2"，其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目

    :return:
    """
    input = './conv_ins.csv'
    # test_input = '/home/mi/blueplan/data/round1_ijcai_18_test_a_20180301.txt'
    # input = '/home/mi/blueplan/data/h1000.txt'
    output = './seminar.string.sample'

    logging.info('begin to make string sample')
    df = pd.read_csv(input,sep=' ',header=0)

    # test_df = pd.read_csv(test_input,sep=' ', header=0)
    # test_df['is_trade'] = -1
    #
    # cdf = pd.concat([df, test_df])
    # print df.shape , test_df.shape , cdf.shape

    cdf = df
    cdf['delivery_bin'] = pd.qcut( cdf['shop_score_delivery'] , np.linspace(0,1 ,11))

    cdf['delivery_bin_fac'] = pd.factorize(cdf['delivery_bin'], sort=True)[0]


    cdf['review_bin'] = pd.qcut( cdf['shop_review_positive_rate'] , np.linspace(0,1 ,11), duplicates='drop')
    cdf['review_bin_fac'] = pd.factorize(cdf['review_bin'], sort=True)[0]

    cdf['description_bin'] = pd.qcut( cdf['shop_score_description'] , np.linspace(0,1 ,11), duplicates='drop')
    cdf['description_bin_fac'] = pd.factorize(cdf['description_bin'], sort=True)[0]

    check_map = {}
    with open(output, 'w') as f:
        for row in cdf.itertuples():
            buf = []

            date=convdate(row.context_timestamp)
            hour = convhour(row.context_timestamp)
            # When add meta ,you must add metanum!!  Must with \t

            finger = "{}_{}".format(row.user_id, row.item_id)
            # if  finger in check_map and check_map[finger] != date:
            #     print finger, date , check_map[finger]
            # else:
            #     check_map[finger] = date
            # 使用  insid ,方便后续预测时的一些.

            metabuf = [
                'uid={insid}'.format(insid=row.instance_id),
                'date={date}'.format(date=date),
                'user_id={uid}'.format(uid= row.user_id),
                'instance_id={}'.format(row.instance_id),
                'context_timestamp={}'.format(row.context_timestamp),
            ]



            metanum = len(metabuf)
            meta = '\t'.join(metabuf)
            buf.append('{metanum}\t{meta}\t{Y}'.format(metanum = metanum , meta= meta, Y=row.is_trade )) # insid is uid now for compatible.

            buf.append('item_id={}'.format(row.item_id))

            buf.extend( conv_item_category_list('item_category', hour, row.item_category_list))

            buf.extend( conv_item_property_list('item_property', row.item_property_list))


            buf.append('item_brand_id={}'.format(row.item_brand_id))
            buf.append('item_city_id={}'.format(row.item_city_id))

            buf.append('item_price_level={}'.format(row.item_price_level))
            buf.append('item_sales_level={}'.format(row.item_sales_level))
            buf.append('item_collected_level={}'.format(row.item_collected_level))
            buf.append('item_pv_level={}'.format(row.item_pv_level))


            # ================= 覆盖度过滤之后 才能直接使用user_id
            buf.append('user_id={}'.format(row.user_id))
            buf.append('user_gender_id={}'.format(row.user_gender_id))
            buf.append('user_age_level={}'.format(row.user_age_level))
            buf.append('user_occupation_id={}'.format(row.user_occupation_id))
            buf.append('user_star_level={}'.format(row.user_star_level))

            buf.append('context_page_id={}'.format(row.context_page_id))
            buf.append('context_hour={}'.format(int(hour)))

            #=============== corss user and item

            buf.append("gender#item={iv}#{jv}".format(iv=row.user_gender_id, jv=row.item_id))
            buf.append("gender#brand={iv}#{jv}".format(iv=row.user_gender_id, jv=row.item_brand_id))
            buf.append("gender#shop={iv}#{jv}".format(iv=row.user_gender_id, jv=row.shop_id))
            buf.append("gender#sales={iv}#{jv}".format(iv=row.user_gender_id, jv=row.item_sales_level))
            buf.append("gender#price={iv}#{jv}".format(iv=row.user_gender_id, jv=row.item_price_level))
            buf.append("gender#collected={iv}#{jv}".format(iv=row.user_gender_id, jv=row.item_collected_level))
            buf.append("gender#pv={iv}#{jv}".format(iv=row.user_gender_id, jv=row.item_pv_level))
            buf.append("gender#delivery={iv}#{jv}".format(iv=row.user_gender_id, jv=row.delivery_bin_fac))


            # cross user and item begin.
            buf.extend(conv_predict_category_property(row.predict_category_property))
            # ======= search category and property correlation
            buf.append('search_category_corr={}'.format(search_category_corr(row)))
            buf.append('search_property_corr={}'.format(search_property_corr(row)))


            buf.append('shop_id={}'.format(row.shop_id))
            buf.append(u'shop_review_num_level={}'.format(row.shop_review_num_level))
            buf.append(u'shop_review_positive_rate={}'.format(row.shop_review_positive_rate))
            buf.append(u'shop_star_level={}'.format(row.shop_star_level))
            buf.append(u'shop_score_service={}'.format(row.shop_score_service))
            buf.append(u'shop_score_delivery={}'.format(row.shop_score_delivery))
            buf.append(u'shop_score_description={}'.format(row.shop_score_description))


            buf.append('item_posrate={}'.format(row.item_posrate))
            buf.append('age_item_posrate={}'.format(row.age_item_posrate))
            buf.append('recent_15minutes={}'.format(row.recent_15minutes))

            f.write('\t'.join(buf) + '\n')



    logging.info('make string sample done')



if __name__ == '__main__':
    fire.Fire()
