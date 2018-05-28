# encoding=utf-8
"""
@author : ZhongQing
"""
import re,logging
import numpy as  np
from collections import defaultdict
from enum import  Enum

FeatureType= Enum("FeatureType" , ('BIN','VAL', 'MIX'))
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
        self.idcovermap = defaultdict(int)
        self.filter_min_cover = 20
        self.valmap = {}  #value map
        self.prefix = prefix
        self.type = type
        self.kwargs = kwargs
        if 'drop' in kwargs:
            self.drop = kwargs['drop']
        else:
            self.drop = False

        if self.type == FeatureType.VAL:
            self.idmap[prefix] = 1

        if self.type == FeatureType.MIX:
            self.idmap[prefix] = 1 #reserve the numerical.
            self.idcovermap[prefix] = 10000000

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
        cnt = self.end_feaid - self.start_feaid
        return "[{0} , {1})  {2} {3}".format( self.start_feaid,self.end_feaid , self.kwargs['name'], cnt )
    def _filterFeature(self):
        if self.type == FeatureType.BIN:
            for k in self.idcovermap:
                c = self.idcovermap[k]
                if c < self.filter_min_cover and k in self.idmap:
                    logging.info('pop {}, {}'.format(k , c ))
                    self.idmap.pop(k)
            

            
    def alignFeatureID(self,start):
        """
        1. filter feature with too few coverage.
        """
        self.start_feaid = start
        self.end_feaid  = start + len(self.idmap)
        self._filterFeature() 
        return self.end_feaid
    def _transformMix(self,prefix, feastr, val):
        is_num = self._check_numerical(prefix, feastr)
        if not is_num:
            k = feastr
            v = val
            logging.info('mixtype = cate {}'.format(feastr))
        else:
            k = prefix
            v = feastr[len(prefix)+1:]

            logging.info('mixtype = numerical {} v={}'.format(feastr, v))

        if k in self.idmap:
            return "{0}:{1}".format(self.start_feaid + self.idmap[k] - 1, v)
        else:
            return -1


    def transform(self, prefix, feastr,  val = 1 ):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}
        :param feastr:
        :return  -1 skip the feastr.
        """

        # prefix, feaval = re.split(sep, feastr, maxsplit=1)
        if prefix != self.prefix or self.drop == True:
            return -1
        if self.type == FeatureType.MIX:
            return self._transformMix(prefix, feastr, val )
        else:
            if self.type == FeatureType.VAL:
                feastr , val  = feastr.split('=')
                if val == 'nan':
                    # logging.info('hit nan in FeatureType.VAL {}'.format(feastr))
                    val = 0
            if feastr in self.idmap:
                return "{0}:{1}".format(self.start_feaid +  self.idmap[feastr] - 1 , val )
            else:
                return -1

    def tryAdd(self, prefix,  feastr):
        """
        填充feature space!!
        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix , feaval = re.split(sep, feastr, maxsplit=2)

        if self.type == FeatureType.MIX:
            return self._tryAddMix(prefix,feastr)
        else:
            if self.drop: return False
            if self.type != FeatureType.BIN : return False
            if prefix == self.prefix:
                if feastr in self.idmap:
                    self.idcovermap[feastr] = self.idcovermap[feastr] + 1  # update feature cover count .
                else:
                    self.idmap[feastr] =len(self.idmap) + 1  # feature encoding !
                    self.idcovermap[feastr] = 1
                return True
            else:
                return False
    def _check_numerical(self,prefix, feastr):
        val =  feastr[len(prefix)+1:]
        try:
            val = float(val)
            return True
        except:
            return False

    def _tryAddMix(self,prefix, feastr):
        """
        需要判断当前是 cate or numerical .
        :param prefix:
        :param feastr:
        :return:
        """
        if prefix == self.prefix:

            is_num = self._check_numerical(prefix, feastr)
            if not is_num:
                if feastr in self.idmap:
                    self.idcovermap[feastr] = self.idcovermap[feastr] + 1
                else:
                    self.idmap[feastr] = len(self.idmap) + 1
                    self.idcovermap[feastr] = 1
            return True
        return False
