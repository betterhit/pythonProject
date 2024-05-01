import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
top_folder = '../bugdata3'  # 替换为你的顶层文件夹路径

# 初始化一个二维列表来存储所有CSV文件的中值
all_medians = []

# 递归函数来处理文件夹


def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".csv"):
                # 构建完整的文件路径
                file_path = os.path.join(root, filename)

                # 读取CSV文件
                df = pd.read_csv(file_path)
                means = df.iloc[:, 1:21].mean().values
                stds = df.iloc[:, 1:21].std().values

                # 构建结果行，包括文件名和40个均值和标准差
                # result_row = list(means) + list(stds)
                # result_row.insert(0, filename)
                # # 将结果行添加到列表中
                # all_medians.append(result_row)
                # 对1到21列进行 z-score 标准化
                columns_to_standardize = df.columns[1:21]
                scaler = StandardScaler()
                df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

                # 计算每一列的中值
                medians = df.iloc[:, 1:21].median().tolist()

                # 将中值列表添加到二维列表
                medians.insert(0, filename)
                all_medians.append(medians)

# 调用递归函数来处理顶层文件夹
process_folder(top_folder)
# for i in all_medians:
#     print(i)
import numpy as np

# all_medians 是包含每个项目中值的二维列表

# 读取数据文件
file_path = "output_auc_spe+.csv"
import numpy as np
from itertools import combinations



def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1[1:]) - np.array(vector2[1:]))

# 计算两两项目之间的欧氏距离
project_distances = []
for pair in combinations(all_medians, 2):
    project1, project2 = pair
    distance = euclidean_distance(project1, project2)
    project_distances.append([project1[0], project2[0], distance])

# 显示结果
# for distance in project_distances:
#     print(distance)
df = pd.read_csv(file_path).iloc[:,0:3]
projectlist = df.values.tolist()
distname = []
for line in projectlist:
        dist=[]
        target = line[0]
        sourcelist = line[1:]
        minscore = 10
        finalsource = ''
        for source in sourcelist:
            str_t = target.split('-')[0]
            str_s = source.split('-')[0]
            for score in project_distances:
                if ((target==score[0] and source==score[1]) or (target==score[1] and source==score[0]))  and str_s!=str_t:
                    # minscore = min(minscore,float(score[2]))
                    # finalsource = source
                    if minscore > float(score[2]):
                        minscore = min(minscore,float(score[2]))
                        finalsource = source
        dist.append(target)
        dist.append(finalsource)
        distname.append(dist)
for i in distname:
    print(i)
df = pd.DataFrame(np.array(distname))
df.to_csv("sort_auc_spe+1.csv",index=False)