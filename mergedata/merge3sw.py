import pandas as pd
import os
import shutil

# 读取csv文件
with open("../singlesource/SPE+/output_final_f1.csv", "r") as file:
    lines = file.readlines()

# 创建保存复制文件的根文件夹
root_folder = "dataf1_4"
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

# 遍历每一行
for line in lines:
    # 分割行并获取文件名列表和目标文件名
    parts = line.strip().split(",")
    target_filename = parts[0]
    filenames = parts[1:5]  # 仅获取第二列到第五列的内容

    # 遍历每个文件名，复制文件内容到以文件名命名的文件夹中
    for filename in filenames:
        filepath = os.path.join("../bugdata3", filename.strip())
        if os.path.exists(filepath):
            # 创建文件夹，以文件名命名
            folder_name = target_filename[0:-4]  # 获取文件名（去掉扩展名）
            target_folder = os.path.join(root_folder, folder_name)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # 复制文件到目标文件夹中
            shutil.copy(filepath, os.path.join(target_folder, filename.strip()))
