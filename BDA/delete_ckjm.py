import os

def delete_ckjm_csv(folder_path):
    # 遍历文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 检查当前文件夹中是否有名为 'ckjm.csv' 的文件
        if 'ckjm.csv' in files:
            # 构建完整路径并删除文件
            file_path = os.path.join(root, 'ckjm.csv')
            os.remove(file_path)
            print(f'已删除文件: {file_path}')

# 要删除文件的文件夹路径
folder_path = 'C:\\Users\\fuhang\Desktop\\3sw'

# 调用函数删除文件夹下所有子文件夹中的 'ckjm.csv' 文件
delete_ckjm_csv(folder_path)