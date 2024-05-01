import pandas as pd
import os

# 获取当前目录及子文件夹下所有的csv文件
csv_files = []
for root, dirs, files in os.walk('.'):  # 搜索当前目录及其子文件夹下的所有文件
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# 读取包含文件名信息的CSV文件
csv_file = 'output_sorted_keys.csv'  # 请将'your_csv_file.csv'替换为实际的文件名和路径

# 读取CSV文件内容并处理
data = pd.read_csv(csv_file)

for index, row in data.iterrows():
    merge_filename = row['0']
    merge_files = [row['1'], row['2'],row['3'],row['4'],row['5']]

    dfs_to_merge = []
    for file in merge_files:
        # 在当前文件夹及其子文件夹下搜索对应的文件
        found_files = [f for f in csv_files if os.path.basename(f) == file]
        if found_files:
            try:
                df = pd.read_csv(found_files[0])
                dfs_to_merge.append(df)
            except Exception as e:
                print(f"读取文件 '{file}' 时发生错误: {e}")
        else:
            print(f"文件 '{file}' 不存在，跳过该文件的合并.")

    if dfs_to_merge:
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)

        # 保存合并后的文件
        merged_df.to_csv(merge_filename, index=False)
        print(f"合并后的文件 '{merge_filename}' 已保存.")
    else:
        print(f"无法合并文件 '{merge_filename}'，因为未找到任何待合并的文件.")
