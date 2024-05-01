import pandas as pd
from scipy.stats import wilcoxon
import scipy.stats as stats

# 读取CSV文件
df = pd.read_csv('../RQ1/speknn_f1.CSV')

# 获取第一列的数据
column1 = df.iloc[:, 0].values  # 假设第一列的索引为1
print(column1)
# 遍历其他列进行符号秩检验
for column in df.columns[1:]:  # 假设第一列之后的列是要进行检验的列
    column_data = df[column].values
    # print(column_data)
    statistic, p_value = stats.wilcoxon(column1, column_data,correction=True, alternative='greater')
    print(f"第一列与{column}列的p值:", p_value)