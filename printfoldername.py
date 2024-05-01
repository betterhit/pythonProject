import os


def list_files_in_directory(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist.")
        return

    # 获取文件夹下的所有文件名
    files = os.listdir(folder_path)

    # 打印文件名
    print(f"Files in directory {folder_path}:")
    for file in files:
        print(file)


# 调用函数并传入文件夹路径
folder_path = 'TCAALL'  # 替换为你的文件夹路径
list_files_in_directory(folder_path)
