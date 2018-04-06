# 最佳实践入门

核心概念： Pipeline

原始数据 到 最终的模型，需要经过一系列转换，这些转换前后相连，形成一个Pipeline 

这里演示了 最常见的一个Pipeline

Scaler -> 例如:  MaxAbsScaler / MinMaxScaler / StandardScaler  

SelectKBest -> 特征选择 

Estimator -> 这个就是常说的 模型，例如LogisticRegressor,lightgbm.LGBMRegressor
