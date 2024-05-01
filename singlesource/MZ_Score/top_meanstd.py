# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:29:26 2023

@author: fuhang
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 指定主文件夹路径
main_folder_path = "../../bugdata3"

# 初始化一个列表，用于存储所有结果
result_list = []

# 遍历主文件夹下的所有子文件夹和CSV文件
for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file.endswith('.csv'):
            # 构建CSV文件的完整路径
            file_path = os.path.join(root, file)
            
            # 读取CSV文件为DataFrame
            df = pd.read_csv(file_path)
            columns_to_standardize = df.columns[1:21]
            scaler = StandardScaler()
            df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
            means = df.iloc[:, 1:21].median().tolist()
            # stds = df.iloc[:, 1:21].std().values
            
            # 构建结果行，包括文件名和40个均值和标准差
            # result_row = [file] + list(means) + list(stds)
            result_row = [file] + list(means)
            # 将结果行添加到列表中
            result_list.append(result_row)

# 构建列名
# columns = ['File'] + [f'Feature{i}_Mean' for i in range(1, 21)] + [f'Feature{i}_Std' for i in range(1, 21)]
columns = ['File'] + [f'Feature{i}_Mean' for i in range(1, 21)]
# 创建结果DataFrame
result_df = pd.DataFrame(result_list, columns=columns)

# 保存结果为CSV文件
result_csv_path = 'result.csv'
result_df.to_csv(result_csv_path, index=False)

# 打印结果文件保存路径
print(f'Results saved to {result_csv_path}')
