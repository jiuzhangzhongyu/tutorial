# 使用说明 
创建目录 data
下载数据,放到data下面
链接: https://pan.baidu.com/s/1KSFeWtBI1GCi8Q3LEJ_-NA 密码: z3g4  

## 预处理生成转化率和实时特征
python preprocess.py    
产出文件： conv_ins.csv

## 生成libsvm格式文件

python  make_sample.py 

输入： conv_ins.csv 
输出: final.sample 

## 训练
python  train.py lgbcv --sample=final.sample 

观察下cross validation的结果 

# 课后习题

加入更多的特征，使得最终的logloss降低到0.88附近 
