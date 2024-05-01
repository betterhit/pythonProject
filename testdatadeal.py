import os
import pandas as pd

# 指定文件夹路径
folder_path = 'testdata3'  # 将 'your_folder_path' 替换为你的文件夹路径

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 循环处理每个CSV文件
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 添加一列并填充为0
    df['42'] = 0

    # 保存修改后的CSV文件
    df.to_csv(file_path, index=False)

print("处理完成")
