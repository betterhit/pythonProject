import csv
import os
import shutil

# 读取1.csv文件
csv_file_path = 'top5folder.csv'

with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')

    # 遍历每一行
    for row in csv_reader:
        # 第一列作为子文件夹名
        subfolder_name = row[0][0:-4]

        # 创建子文件夹
        subfolder_path = os.path.join('top5data', subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # 遍历除第一列之外的列
        for csv_filename in row[1:]:
            # 构建完整的CSV文件路径
            csv_file_path = os.path.join('top5data', csv_filename)

            # 检查文件是否存在
            if os.path.isfile(csv_file_path):
                # 复制文件到子文件夹中
                shutil.copy(csv_file_path, subfolder_path)
            else:
                print(f"File not found: {csv_filename} in row {row}")
