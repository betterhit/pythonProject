# 读取原始内容
with open('final_auc.txt', 'r') as f:
    original_data = f.readlines()

# 处理每行数据
new_data = []
for line in original_data:
    line = line.strip().split(', ')  # 分割每行数据
    file_names = line[:2]  # 前两个元素为文件名
    feature_indices = line[2:]  # 后面的元素为特征索引
    new_line = ','.join(file_names) + ',' + ','.join(map(str, feature_indices))  # 合并为新行
    new_data.append(new_line)

# 将新数据保存到文件中
with open('feature_auc.txt', 'w') as f:
    for line in new_data:
        # 去除每行数据的外侧方括号和单引号并去除最后一个元素后面的逗号，然后写入文件
        line = line.strip('[]').replace("'", "")
        line = line.rstrip(',')
        f.write("%s\n" % line)
