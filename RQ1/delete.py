import os
import pandas as pd

# 文件夹路径
folder_path = 'C:/Users/fuhang/Desktop/实验数据统计/RQ1 迁移学习调查/AUC'

# 获取文件夹中的所有 CSV 文件
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 遍历每个 CSV 文件
for file in csv_files:
    # 构建完整文件路径
    file_path = os.path.join(folder_path, file)

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 删除 result 列
    if 'result' in df.columns:
        df = df.drop(columns=['result'])

    # 将修改后的 DataFrame 保存回原文件
    df.to_csv(file_path, index=False)
