import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
import joblib

# 从训练集CSV文件中加载数据
train_data = pd.read_csv("../bugdata3/ant-1.3.csv")

# 拆分特征和目标
X_train = train_data.iloc[:, 1:20]
y_train = train_data.iloc[:, 21]

# 标准化特征数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 建立SVR模型
svr = SVR(kernel='linear')

# 进行CFS特征选择
feature_selector = SequentialFeatureSelector(svr, scoring='neg_mean_squared_error', n_features_to_select=10)
feature_selector.fit(X_train_scaled, y_train)

# 获取选定的特征列索引
selected_features_indices = feature_selector.get_support(indices=True)

# 选取训练数据中对应选定特征的列
X_train_selected = X_train.iloc[:, selected_features_indices]
feature_names = X_train.columns

# 打印选定特征的原始列序号和列名
for idx in selected_features_indices:
    print(f"Original Column Index: {idx}, Original Column Name: {feature_names[idx]}")
# 训练模型
svr.fit(X_train_selected, y_train)

# 保存模型到文件
joblib.dump(svr, 'svr_model_with_cfs.pkl')
