# pandas 核心操作
```python
import pandas as pd 
pd.read_csv 

# [推荐这种遍历方式]
for row in df.itertuples(): 
# 另一种遍历
for index, row in df.iterrows(): 

# 数据转换，利用旧的列生成新的一列
df['new_column'] = df['column'].apply(lambda x : function(x))  : 
#  group by  then  sum or count 操作,最后的reset_index 重置 index,可观察下使用和不使用的区别 
df.groupby('column_name').column.sum().reset_index() :
#df1,df2合并成一个 DataFrame，可以按照行合并(axis=0) ,也可以按照列合并(axis =1 ) 同一条样本的多组特征就可以按照列并起来

pd.concat ( [df1, df2] ,axis= 0 )  
# 按照双方id相等 的条件 关联起来； how='left'类似于 sql中的左连接，其他连接方式，自行查阅 
pd.merge( df1, df2, on='id',how='left') 
```
# quantile 分位点
## 内涵解释
给定一个数组，假设包括100个数字， 我们要把这些数按大小 等分成10堆， 每堆就有 10个数， 这个堆里的最大数，就是本堆的 右边界 ; 所有堆的右边界都找出来，就形成了 整个数组的 10分位数；  （10分位 就是指分成 10个堆的意思 ） 

先进行排序，之后按照顺序挨个取 10个数 ， 记录下最大的数字，就是该堆的 右边界 ； 逐次把 每个堆的 右边界都找出来 
如下： 10 ， 20 分别就是 前 两个堆的右边界   
1 2  3 4  5 6 7 8 9 *10* ;  11 12 13 14 15 16 17 18 19 *20* ;  .... 

考虑一种特殊情况:  前20个数字都一样， 这样我们就可以把 前两个堆合并成一个 大堆 ，最后得到的是 9 个堆  

## 代码操作 
```python
import numpy as np
import pandas as pd
#参数解释
#q: np.linspace(0,1,11) 返回的就是 [0 , 0.1 ,0.2 ,..., 1]  linspace 就是等分的意思 ; 
# 这里的 0.1,0.2代表的是 累计个数占比； 这样第一个堆个数占比就是  0.1 - 0  ；
# 第二个堆个数单独占比就是 0.2 -0.1 = 0.1   ； 
#  手工输入的话，可以让每个堆的个数占比不一样，例如4个堆，占比如 [0,0.4,0.5,0.53,1] 
#retbins: 返回每堆的右边界 
#duplicates: 遇到两个堆的右边界相等的情况，如何处理?  这里的drop表示 丢掉 其中一个右边界，本质上就是把两个堆 并成一个堆了 
c,  bins = pd.qcut( df['price'], q = np.linspace(0,1,11) ,retbins=True,  duplicates='drop') 
```
# outlier 离群点 
## 直观概念
给定一个数组，有些特别大 或者特别小的数字，这些数字很可能是异常值 ,我们需要定位到异常值； 并且进行一些修复 ，这是机器学习预处理的重要步骤
## 利用四分位数处理离群点

```python
# 利用describe直接得到四分位数信息；当然也可以利用 上面的qcut得到四分位数 
des = df['price'].describe()
# 计算出允许的最大值 系数3是经验数值，可以自己调整
valid_max =  des['50%'] + 3* ( des['75%'] -des['50%'] ) 
# 进行clip ； clip就是约束 最大最小值  
df['price'] = df['price'].clip(0, valid_max)  
#到此，我们就把price的离群点修正了 
```
## 利用 均值/标准差 处理离群点 
```python
#均值
m = df['price'].mean() 
#标准差
s = df['price'].std()
# 近似处理： 正态分布 [mean - 3 *std , mean + 3 * std] 这个范围内的 累计分布概率是 99%以上; 这个范围之外的，基本是小概率事件，可认为是离群点
# 对价格来说，我们关注的特别高价格的东西，最低价格是0，也是可以理解的
valid_max = m + 3 * s 
df['price'] = df['price'].clip( 0 , valid_max)  

```

# One Hot Encoding 

OneHotEncoding 是机器学习最常用的特征处理方式 ，pandas提供支持函数  
pd.get_dummies()  

日常使用中，另一种实现方式是:scikit-learn中的LabelEncoder() 和 OneHotEncoder()  ，感兴趣可自行实验  

# Discretize 

连续值离散化是非常有效的特征处理方式 ， 比如说 把浮点数价格转换为 N个价格区间，就是 离散化  

下面两行代码把 列column 离散化变成 column_bin_fac  列  ； 

这里np.linspace(0,1,11) 会生成 十分位点 array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

```python
df['column_bin'] = pd.qcut( df['column'] , np.linspace( 0 ,1 , 11  ) ,duplicates='drop')  

df['column_bin_fac'] = pd.factorize( df['column_bin'], sort=True)  
```
# Check Correlation 检查相关性

检查数据相关性 是机器学习特征识别的 基础工作，在了解数据的基础上才能够针对性的设计出解决方案 

pd.corr(df)  可以一次性输出df 所有列的相关系数， 

[corr](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html)

# 课后习题

参见 data/README.md 这份数据在后续广告的课程中也会用到，来自阿里巴巴平台的真实数据
