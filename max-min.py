import pandas as pd

# 读取原始 CSV 文件


import pandas as pd
import numpy as np


import pandas as pd

# 读取原始 CSV 文件
input_file = 'apache.csv'
output_file = 'apache_prc.csv'
data = pd.read_csv(input_file)

# 选择需要归一化的列（第1列到第21列）
columns_to_normalize = data.columns[0:25]

# 使用最小最大归一化处理数据
normalized_data = (data[columns_to_normalize] - data[columns_to_normalize].min()) / (data[columns_to_normalize].max() - data[columns_to_normalize].min())

# 限制小数位数为4位（根据需要进行调整）
normalized_data = normalized_data.round(4)

# 更新原始数据的归一化列
data[columns_to_normalize] = normalized_data

# 保存处理后的数据到新的 CSV 文件
data.to_csv(output_file, index=False)

