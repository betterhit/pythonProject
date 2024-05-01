import csv
import numpy as np

# 从CSV文件读取数据
def read_csv(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # 读取列名
        column_names = next(reader)
        # 读取数据
        data = []
        for row in reader:
            print(row)
            data.append([float(value) for value in row])
    return column_names, np.array(data)

# 定义胜场/平场/负场的判断函数
def compare_values(value1, value2):
    if value1 > value2:
        return '胜场'
    elif value1 < value2:
        return '负场'
    else:
        return '平场'

# 统计胜场/平场/负场的次数
def count_outcomes(column1, column2):
    wins = 0
    draws = 0
    losses = 0
    for value1, value2 in zip(column1, column2):
        outcome = compare_values(value1, value2)
        if outcome == '胜场':
            wins += 1
        elif outcome == '负场':
            losses += 1
        else:
            draws += 1
    return wins, draws, losses

# 读取数据
file_path = 'RQ2 F1.CSV'  # 请替换成你的CSV文件路径
column_names, data = read_csv(file_path)

# 对每一对列进行比较
for i in range(data.shape[1]):
    for j in range(i + 1, data.shape[1]):
        wins, draws, losses = count_outcomes(data[:, i], data[:, j])
        column1_name = column_names[i]
        column2_name = column_names[j]
        print(f'对比 {column1_name} 列与 {column2_name} 列:')
        print(f'胜场: {wins}, 平场: {draws}, 负场: {losses}\n')
