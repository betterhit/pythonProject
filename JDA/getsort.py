import joblib
import pandas as pd
import os

# Load the model
svr = joblib.load('svr_model.pkl')

# 存储所有文件的排序结果的key字段以及文件名
all_files_keys = []

# 遍历测试数据集文件夹中的所有文件
folder_path = "../testdata5/"  # 测试数据集文件夹路径
files = os.listdir(folder_path)

for file in files:
    # 从测试集CSV文件中加载数据
    test_data = pd.read_csv(folder_path + file)
    X_test = test_data.iloc[:, 2:41]

    y_pred = svr.predict(X_test)

    # 获取test_data的第一列值
    test_ids = test_data.iloc[:, 0]  # 假设第一列是标识符或索引

    # 创建字典
    result_dict = dict(zip(test_ids, y_pred))

    # 按y_pred的大小对字典进行排序
    sorted_result = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}
    name = file.split('_')[0]
    # 将文件名插入到排序结果的键列表的最前面
    keys_list = list(sorted_result.keys())
    keys_list.insert(0, name)

    # 将排序结果的key字段添加到二维列表中
    all_files_keys.append(keys_list)

# 将二维列表写入CSV文件
#use knn spe+
output_csv = "output_f1_spe+1.csv"  # 输出CSV文件名
df = pd.DataFrame(all_files_keys)  # 转换为DataFrame
df.to_csv(output_csv, index=False)  # 写入CSV文件
