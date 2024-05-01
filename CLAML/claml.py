import os
import pandas as pd

# 指定文件夹路径
folder_path = "../bugdata3"

# 存储结果的列表
results = []

# 遍历文件夹下的所有csv文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取特征列和标签列的范围
        feature_cols = df.columns[1:21]  # 第二列到第20列是特征列
        label_col = df.columns[21]  # 第22列是标签列

        # 计算每列的中值
        medians = df[feature_cols].median()

        # 初始化违规分数字典
        violations = {i: 0 for i in range(len(feature_cols))}

        # 遍历每一行
        for _, row in df.iterrows():
            label = row[label_col]
            for i, col in enumerate(feature_cols):
                if label == 0:
                    if row[col] > medians[col]:
                        violations[i] += 1
                else:
                    if row[col] <= medians[col]:
                        violations[i] += 1

        # 对违规分数进行排序
        sorted_violations = sorted(violations.items(), key=lambda x: x[1], reverse=True)

        # 获取前十个违规分数的特征列索引
        top_violations = sorted_violations[:20]
        top_features = [str(col + 1) for col, _ in top_violations]  # 将索引转换为字符串格式

        # 将文件名和前十个特征列索引存储到结果列表中
        results.append((file_name, top_features))

# 将结果写入txt文件
output_file_path = "../singlesource/SPE/claml12.txt"
with open(output_file_path, "w") as f:
    for file_name, features in results:
        f.write(f"{file_name},")
        f.write(",".join(features))
        f.write("\n")
