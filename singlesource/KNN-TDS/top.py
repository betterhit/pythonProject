# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:58:41 2023

@author: fuhang
"""

import pandas as pd
import numpy as np

# 加载目标项目的CSV文件
target_file = "bug-data/ant/ant-1.4.csv"
target_df = pd.read_csv(target_file)

# 加载其他项目合并后的CSV文件
merged_file = "ant1.4_merge.csv"
merged_df = pd.read_csv(merged_file)

# 提取特征列（排除第一列和最后一列）
target_features = target_df.iloc[:, 1:-1]
merged_features = merged_df.iloc[:, 1:-1]

# 创建一个空的DataFrame，用于保存筛选后的结果
filtered_df = pd.DataFrame()

# 循环遍历目标项目的每个实例
for index, target_row in target_features.iterrows():
    # 计算与目标实例最相似的实例，使用欧氏距离
    similarity_scores = []
    for _, other_row in merged_features.iterrows():
        # 使用numpy的linalg.norm计算欧氏距离，排除第一列和最后一列
     
        distance = np.linalg.norm(target_row.values - other_row.values)
 
        # 将距离添加到相似性分数列表中
        similarity_scores.append(distance)
     

    # 选择最相似的10个实例
    top_k_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i])[:10]

    # 将找到的最相似的实例添加到结果DataFrame中
    top_k_instances = merged_df.iloc[top_k_indices]
    filtered_df = pd.concat([filtered_df, top_k_instances])

# 保存筛选后的结果为CSV文件
filtered_df = filtered_df.drop_duplicates()
filtered_df.to_csv("filtered_newant1.4.csv", index=False)
