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

        if self.type == FeatureType.VAL:
            self.idmap[prefix] = 1
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
    def transform(self, prefix, feastr,  val = 1 ):
        """
        具体的特征转换成  libsvm 格式.  {id}:{val}
        :param feastr:
        :return  -1 skip the feastr.
        """

        # prefix, feaval = re.split(sep, feastr, maxsplit=1)
        if prefix != self.prefix or self.drop == True:
            return -1

        if self.type == FeatureType.VAL:
            feastr , val  = feastr.split('=')
            if val == 'nan':
                logging.info('hit nan in FeatureType.VAL')
                val = 0
        if feastr in self.idmap:
            return "{0}:{1}".format(self.start_feaid +  self.idmap[feastr] - 1 , val )
        else:
            return -1

    def tryAdd(self, prefix,  feastr):
        """

        :param feastr:
        :param sep:
        :return:  True: 添加成功 False: 添加失败
        """
        # prefix , feaval = re.split(sep, feastr, maxsplit=2)
        if self.drop: return False
        if self.type != FeatureType.BIN: return False
        if prefix == self.prefix:
            if feastr in self.idmap:
                pass
            else:
                self.idmap[feastr] =len(self.idmap) + 1
            return True
        else:
            return False
