# 读取txt1和txt2的内容
with open('cfs.txt', 'r') as f:
    txt1_lines = f.readlines()

with open('gaininfo.txt', 'r') as f:
    txt2_lines = f.readlines()

# 处理txt1的内容，按照合并规则进行合并
txt3_lines = []
for i, line in enumerate(txt1_lines):
    file_name = line.strip().split(',')[0]  # 获取文件名
    txt1_numbers = [int(num) for num in line.strip().split(',')[1:]]  # 忽略文件名部分
    txt2_row = [int(num) for num in txt2_lines[i].strip().split(',')] if i < len(txt2_lines) else []

    # 合并两行数据
    merged_numbers = txt1_numbers[:]

    # 逐个添加txt2中的元素到txt1的末尾，确保不重复
    for num in txt2_row:
        if len(merged_numbers) >= 15:
            break  # 如果已经有11个元素了，则停止添加
        if num not in merged_numbers:
            merged_numbers.append(num)

    # 将文件名和合并后的结果组合成一行
    merged_line = file_name + ',' + ','.join(map(str, merged_numbers))
    txt3_lines.append(merged_line)

# 将合并后的结果写入txt3
with open('txt3.txt', 'w') as f:
    for line in txt3_lines:
        f.write(line + '\n')