import pandas as pd
import numpy as np

# 读取包含项目特征的CSV文件为DataFrame
data = pd.read_csv('result.csv')

# 提取项目名称列
project_names = data['File'].tolist()

# 提取项目特征列，这里假设你的特征列从第二列开始，你可以根据实际情况调整
features = data.iloc[:, 1:21].values

# 计算项目之间的欧氏距离矩阵
distance_matrix = np.linalg.norm(features[:, np.newaxis, :] - features, axis=2)

# 将距离矩阵转换为DataFrame，方便后续操作
distance_df = pd.DataFrame(distance_matrix, columns=project_names, index=project_names)

# 创建一个新的DataFrame来存储每个项目选择的最近的三个项目版本
selected_projects = pd.DataFrame(index=project_names, columns=['1st Closest', '2nd Closest', '3rd Closest'])

# 遍历每个项目，为其选择最近的三个项目版本
for project in project_names:
    # 从距离矩阵中获取当前项目到其他项目的距离
    distances = distance_df[project].copy()
    
    # 排除自身距离（距离为0）
    distances = distances.drop(project)
    
    # 提取项目名中"-"号前的内容
    project_prefix = project.split("-")[0]
    
    # 选择最近的三个项目，并确保它们的前缀不同于目标项目的前缀
    closest_projects = []
    
    while len(closest_projects) < 3:
        # 找到距离最小的项目
        closest_project = distances.idxmin()
        
        # 提取最近项目的前缀
        closest_project_prefix = closest_project.split("-")[0]
        
        # 如果前缀不同于目标项目的前缀，添加到最近项目列表中
        if closest_project_prefix != project_prefix:
            closest_projects.append(closest_project)
        
        # 从距离列表中移除已选择的项目
        distances = distances.drop(closest_project)
        
        # 如果没有找到符合条件的项目，跳出循环
        if len(distances) == 0:
            break
    
    # 如果找不到足够的符合条件的项目，可以考虑采取其他策略或者留空
    while len(closest_projects) < 3:
        closest_projects.append(None)
    
    # 将结果存储到新的DataFrame中
    selected_projects.loc[project] = closest_projects

# 打印最终结果
print(selected_projects)
# 将结果保存到CSV文件
selected_projects.to_csv('selected_projects.csv')

