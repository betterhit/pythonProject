import numpy as np
import pandas as pd
from minepy import MINE  # MINE用于计算MIC值

# 读取CSV文件
data = pd.read_csv('your_file.csv')

# 选择特征列
features = data.iloc[:, 1:21].values  # 假设特征列从第二列到第二十一列

# 计算MIC值
def calculate_MIC(source_features, target_features):
    mic_scores = []
    for source_feature in source_features.T:  # 对于每个源项目特征
        mic = MINE()
        mic.compute_score(source_feature, target_features)
        mic_scores.append(mic.mic())
    return mic_scores

# 聚类特征排序
def sort_features_by_MIC(cluster_indices, source_features, target_features):
    sorted_features_indices = []
    for cluster in cluster_indices:
        cluster_features = source_features[:, cluster]
        mic_scores = calculate_MIC(cluster_features, target_features)
        sorted_indices = np.argsort(mic_scores)[::-1]  # 根据MIC值降序排序
        sorted_features_indices.append(cluster[sorted_indices])
    return sorted_features_indices

# 设置目标项目数据（假设是target_features）
target_features = np.random.rand(100, 20)  # 假设有100个样本和20个特征

# 设置聚类结果（假设是cluster_indices）
cluster_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# 对每一个聚类特征进行排序
sorted_indices = sort_features_by_MIC(cluster_indices, features, target_features)
print("Sorted feature indices for each cluster:")
for i, indices in enumerate(sorted_indices):
    print(f"Cluster {i+1}: {indices}")
