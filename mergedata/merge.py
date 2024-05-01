import pandas as pd
import os

# 读取csv文件
with open("../Tradaboost/output_f1_1.csv", "r") as file:
    lines = file.readlines()

# 遍历每一行
save_folder = "data_tra"

# 遍历每一行
for line in lines:
    # 分割行并获取文件名列表和目标文件名
    parts = line.strip().split(",")
    target_filename = parts[0]
    filenames = parts[1:6]  # 仅获取第二列到第五列的内容
    print(filenames)

    # 初始化合并后的内容
    merged_content = pd.DataFrame()

    # 遍历每个文件名，合并文件内容
    for filename in filenames:
        filepath = os.path.join("../bugdata3", filename.strip())
        if os.path.exists(filepath):
            file_content = pd.read_csv(filepath)
            merged_content = pd.concat([merged_content, file_content], axis=0)  # 使用axis=1进行列合并
        else:
            print(f"File {filename} not found in the specified folder.")
            continue

    # 构建保存文件的路径
    save_filepath = os.path.join(save_folder, target_filename.rsplit(".",1)[0] + "_merge.csv")

    # 将合并后的内容保存为新文件
    merged_content.to_csv(save_filepath, index=False)

    print(f"Merged file saved as {save_filepath}")
