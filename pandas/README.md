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
# Check Correlation

检查数据相关性 是机器学习特征识别的 基础工作，在了解数据的基础上才能够针对性的设计出解决方案 

pd.corr(df)  可以一次性输出df 所有列的相关系数， 

[corr](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html)

# 课后习题

参见 data/README.md 这份数据在后续广告的课程中也会用到，来自阿里巴巴平台的真实数据
