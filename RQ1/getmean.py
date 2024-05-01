import os
import pandas as pd

# 文件夹路径
folder_path = '../TCA+'

# 获取文件夹中的所有 CSV 文件
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 遍历每个 CSV 文件
for file in csv_files:
    # 构建完整文件路径
    file_path = os.path.join(folder_path, file)

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 计算每行的平均值和标准差
    means = df.mean(axis=1)
    stds = df.std(axis=1)

    # 将平均值和标准差连接起来，并以+-号分隔
    # result = means.astype(str) + '+-' + stds.astype(str)
    result = means
    # 将结果添加到 DataFrame 的最后一列
    df['result'] = result

    # 将修改后的 DataFrame 保存回原文件
    df.to_csv(file_path, index=False)
