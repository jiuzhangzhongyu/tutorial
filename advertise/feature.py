# encoding=utf-8
"""
@author : zhongqing
"""
import re,logging
import numpy as  np

from enum import  Enum

FeatureType= Enum("FeatureType" , ('BIN','VAL'))
class Feature(object):
    def __init__(self, prefix, startid, type, **kwargs ):
        """
        这里的Feature 代指 特征类别
        特征空间要素： 覆盖范围 特征含义

        统计要素单独处理

        处理场景: 普通特征+ 交叉特征
        """
        self.start_feaid , self.end_feaid = startid, startid
        self.idmap = {}
        self.valmap = {}  #value map
        self.prefix = prefix
        self.type = type
        self.kwargs = kwargs
        if 'drop' in kwargs:
            self.drop = kwargs['drop']
        else:
            self.drop = False
        pass
    def name(self):
        """
        特征含义
        :return:
        """
        return self.kwargs['name']

    def getIdMap(self):
        return self.idmap
    def coverRange(self):
        """
        feature id 的覆盖范围
        [s,e)
        :return:
        """
        return "[{0} , {1})  {2}".format( self.start_feaid,self.end_feaid , self.kwargs['name'])
    def alignFeatureID(self,start):

        self.start_feaid = start
        self.end_feaid  = start + len(self.idmap)
        return self.end_feaid
    def transform(self, prefix, feastr,sep =':', val = 1 ):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}
        :param feastr:
        :return  -1 skip the feastr.
        """

        # prefix, feaval = re.split(sep, feastr, maxsplit=1)
        if prefix != self.prefix or self.drop == True:
            return -1

        if self.type == FeatureType.VAL:
            feastr , val  = feastr.split(':')
            if val == 'nan':
                logging.info('hit nan in FeatureType.VAL')
                val = 0
        if feastr in self.idmap:
            return "{0}:{1}".format(self.start_feaid +  self.idmap[feastr] - 1 , val )
        else:
            return -1

    def tryAdd(self, prefix,  feastr, sep=':'):
        """

        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix , feaval = re.split(sep, feastr, maxsplit=2)
        if self.drop: return False
        if prefix == self.prefix:
            if feastr in self.idmap:
                pass
            else:
                self.idmap[feastr] =len(self.idmap) + 1

            return True
        else:
            return False


class StatFeature(Feature):

    def __init__(self, prefix , startid,expand =False, **kwargs  ):
        """
        统计特征
        单维度统计特征+ 交叉统计特征 ;

        只会是一种情况的统计特征  , 多维度统计的另外考虑

        NOTE: 保证测试集 和 训练集中的ID 一致
        :param prefix:
        :param start_id:
        :param expand: True - 在特征级别 统计以后，再编码成 统计特征
        """


        start_id = startid
        self.start_feaid = start_id
        self.end_feaid = start_id +1

        self.idmap = {}
        self.expand = expand
        self.valmap = {}  # value map
        self.prefix = prefix
        self.kwargs= kwargs
        if 'name' in kwargs:
            self.kwargs['name'] = 'stat_{0}'.format(self.kwargs['name'])
        if 'drop' in kwargs:
            self.drop = kwargs['drop']
        else:
            self.drop = False

        if 'default' in kwargs:
            self.default = kwargs['default']
        else:
            self.default = None

        if not self.drop and self.expand  and  'idfile' in self.kwargs:

            self.loadIdMap_(self.kwargs['idfile'])


        pass

    def name(self):
        """
        特征含义
        :return:
        """
        return self.kwargs['name']
    def coverFeaId(self,feaid):
        if not self.drop and feaid < self.end_feaid and feaid >= self.start_feaid: 
            return True
        return False
    def alignFeatureID(self, start):

        self.start_feaid = start
        if self.expand:
            self.end_feaid = start + len(self.idmap)
        else:
            self.end_feaid = start +1
        return self.end_feaid
    def transform(self, prefix, feastr ,sep=":"):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}
        :param feastr:
        :return  -1 skip the feastr.
        """

        # prefix, feaval = re.split(sep, feastr, maxsplit=1)
        if prefix != self.prefix or self.drop:
            return -1

        if feastr in self.valmap:
            if self.expand:
                month , feaval = re.split(sep,feastr,maxsplit=1)
                if feaval in self.idmap:
                    return "{0}:{1}".format(self.idmap[feaval] + self.start_feaid -1 , self.valmap[feastr])
                else:
                    return -1

            else:
                return "{0}:{1}".format(self.start_feaid ,  self.valmap[feastr])
        else:
            if self.default is not None and self.expand == False:
                return "{0}:{1}".format(self.start_feaid, self.default)
            else:
                return -1
    def loadIdMap_(self,file):
        """
        加载一个idmap，保证训练集和测试集是一致的 feature id
        :return:
        """
        with open(file ,'r') as f:
            for L in f:
                feaval = L.strip()
                prefix, val = feaval.split(':')
                if prefix == self.prefix and  feaval not in self.idmap:
                    self.idmap[feaval] = len(self.idmap) + 1
        logging.info('done loadidmap '+ file )



    def tryAdd(self, prefix, feastr, val , sep=':'):
        """

        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix, feaval = re.split(sep, feastr, maxsplit=2)
        if self.drop: return False
        if prefix == self.prefix:
            if feastr in self.valmap:
                pass
            else:
                self.valmap[feastr] = val
                if self.expand:
                    month, feaval = re.split(sep, feastr, maxsplit=1)

                    if feaval not in self.idmap:
                        self.idmap[feaval] = len(self.idmap) +1
            return True
        else:
            return False


class SuperStatFeature(Feature):
    def __init__(self, prefix, startid, cnt , **kwargs):
        """
        统计特征
        其中某个维度是 数组的情况

        Top-n Feature
        """

        start_id = startid
        self.start_feaid = start_id
        self.end_feaid = start_id + cnt

        self.valmap = {}  # value map
        self.prefix = prefix
        self.cnt =  cnt
        self.kwargs = kwargs

        pass

    def name(self):
        """
        特征含义
        :return:
        """
        return "name"


    def alignFeatureID(self, start):

        self.start_feaid = start
        self.end_feaid = start + self.cnt
        return self.end_feaid


    def transform(self, prefix, feastr_list, sep=":"):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}

        选统计值Top-cnt的作为特征 ，保持特征空间可控


        :param feastr:
        :return  -1 skip the feastr.
        """

        if type(feastr_list) != list:
            return -1
        if prefix != self.prefix:
            return -1
        buf = []
        for feastr in feastr_list:
            if feastr in self.valmap:
                buf.append( float(self.valmap[feastr]))

        buf = sorted(buf, reverse=True)

        strbuf = []

        i = 0
        for v in buf[:self.cnt]:
            strbuf.append("{0}:{1}".format( self.start_feaid + i , v  ))
            i +=1

        if len(strbuf) == 0:
            return -1
        else:
            return ' '.join(strbuf)


    def tryAdd(self, prefix,  feastr, val, sep=':'):
        """

        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix, feaval = re.split(sep, feastr, maxsplit=2)

        if prefix == self.prefix:
            if feastr in self.valmap:
                pass
            else:
                self.valmap[feastr] = val
            return True
        else:
            return False

def initFeatureList():
    name = Feature(prefix='name',startid= 1 , name= 'name')
    age = Feature(prefix='age', startid=1,name = 'age')
    gender = Feature(prefix='gender', startid=1,name = 'gender')
    age_gender = Feature(prefix='age_gender', startid=1,name = 'age_gender' )


    return [name,age,gender, age_gender]

def initStatFeatureList():
    name = StatFeature(prefix='name', startid=1,name = 'name-stat')
    age = StatFeature(prefix='age', startid=1, name = 'age-stat' )
    gender = StatFeature(prefix='gender', startid=1, name = 'gender-stat')
    age_gender = StatFeature(prefix='age_gender', startid=1 ,name='age-gender-stat')

    name_e = StatFeature(prefix='name', startid=1,expand=True, name= 'exp')
    age_e = StatFeature(prefix='age', startid=1,expand=True,name = 'age-exp')
    gender_e = StatFeature(prefix='gender', startid=1,expand=True,name = 'gender-exp')
    age_gender_e = StatFeature(prefix='age_gender', startid=1,expand=True, name ='age-gender-exp')

    return [name,age,gender,age_gender,name_e, age_e, gender_e, age_gender_e]


def initSuperStatFeatureList():
    hobby = SuperStatFeature(prefix='hobby', startid=1,cnt = 3 , name = 'super-hobby')
    return [hobby]

def main():
    """
    make an example
    :return:
    """
    fealist = initFeatureList()
    statfealist = initStatFeatureList()
    superstatfealist = initSuperStatFeatureList()

    idcache = 'data/idcache.txt'
    with open(idcache, 'r') as f:
        for L in f:
            feastr, cover = re.split("\\s+", L , maxsplit=1)
            cover = int(cover)

            if cover >= 1000 : # cover threshold
                for fea in fealist:
                    o = fea #type: Feature
                    o.tryAdd(feastr)

    idcache = 'data/idstat.txt'
    with open(idcache, 'r') as f:
        for L in f:
            feastr, cover,  stat = re.split("\\s+", L, maxsplit=2)
            cover = int(cover)
            stat = float(stat)
            if cover >= 1000:  # cover threshold
                for fea in statfealist:
                    o = fea  # type: StatFeature
                    o.tryAdd(feastr, stat)

                for fea in superstatfealist:
                    fea.tryAdd(feastr, stat)

    start = 1
    for obj in fealist:
        start = obj.alignFeatureID(start)

    for obj in statfealist:
        start = obj.alignFeatureID(start)

    for obj in superstatfealist:
        start = obj.alignFeatureID(start)



    buf =[ fealist , statfealist, superstatfealist ]

    for i in buf:
        for j in i:
            fea = j #type:Feature
            print fea.coverRange()


    #now start is the maximum feature-id
    print '-' * 10
    fbuf = ['name:n1' , 'name:n2' , 'age:2' , 'age:3' , 'age_gender:1_1' ]
    for fea in fbuf:
        for feaobj in buf:
            for f in feaobj:
                x = f.transform(fea)
                if x != -1 :
                    print f.coverRange() ,  x

    fbuf = ['hobby:game' ,'hobby:game1' , 'hobby:game2' ,'hobby:game3']


    for fea in superstatfealist:
        print fea.transform(fbuf)




if __name__ == '__main__':
    main()


