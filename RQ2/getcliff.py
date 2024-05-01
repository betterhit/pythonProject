import pandas as pd

# 读取CSV文件
df = pd.read_csv('RQ2_AUC.csv')

# 获取第一列的数据
column1 = df.iloc[:, 0].values

# 遍历其他列进行Cliff's delta计算
for column in df.columns[1:]:
    column_data = df[column].values

    # 计算大于和小于关系的数量
    n1 = sum(column1 > column_data)
    n2 = sum(column1 < column_data)

    # 计算相等关系的数量
    n_equal = sum(column1 == column_data)

    # 计算Cliff's delta值
    N = len(column1)
    delta = (n1 - n2) / (N - n_equal)

    # 输出Cliff's delta值
    print(f"第一列与{column}列的Cliff's delta值:", delta)
