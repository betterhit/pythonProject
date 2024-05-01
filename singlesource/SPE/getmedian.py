import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# 指定顶层文件夹路径
top_folder = '../../bugdata3'  # 替换为你的顶层文件夹路径

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
                # columns_to_standardize = df.columns[1:21]
                # scaler = StandardScaler()
                # df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
                medians = df.iloc[:, 1:21].median().tolist()


                # df = pd.read_csv(file_path)
                # columns_to_standardize = df.columns[1:21]
                # scaler = MinMaxScaler()
                # normalized_data = scaler.fit_transform(df[columns_to_standardize])
                # df[columns_to_standardize] = normalized_data
                # medians = df.iloc[:, 1:21].median().tolist()


                # df = pd.read_csv(file_path)
                # columns_to_standardize = df.columns[1:21]
                # scaler = Normalizer()
                # normalized_data = scaler.fit_transform(df[columns_to_standardize])
                # df[columns_to_standardize] = normalized_data
                # medians = df.iloc[:, 1:21].median().tolist()



                # 计算每一列的中值
                # medians = df.iloc[:, 1:21].median().tolist()
                # random_row = df.sample(n=1, random_state=42).values.tolist()
                # 计算所选行的最大值
                # medians = random_row[0][1:21]
                # print(medians)
                # 将文件名添加到中值列表的第一个位置
                medians.insert(0, filename)

                # 将中值列表添加到二维列表
                all_medians.append(medians)

# 调用递归函数来处理顶层文件夹
process_folder(top_folder)

# 打印或进一步处理all_medians
# for medians in all_medians:
#     print(medians)
file=open('adm_f1_new.txt',mode='r',encoding='UTF-8')
admin=[]
contents = file.readlines()

for msg in contents:
    msg = msg.strip('\n')

    adm = msg.split()
    # adm = [item for item in adm if item.strip()]

    admin.append(adm)
file.close()
# for i in all_medians:
#     print(i)
def getmodel(admin,all_medians):
    lists = []
    for i in admin:
       k=0
       list1=[]
       list2=[]
       list3=[]
       name1=''
       name2=''
       for j in all_medians:

           if i[1] == j[0] :
               name1=j[0]
               k=k+1
               list1+=j[1:]
           if i[0] == j[0]:
               name2=j[0]
               k=k+1
               list2 += j[1:]
           if k == 2 :
               # print(name1)
               list3.append(name1)
               list3.append(name2)
               list3 += list1
               list3 += list2
               list3.append(i[2])
               lists.append(list3)

               break
    # for i in lists:
    #     print(i)
    # 转换为Pandas DataFrame
    df = pd.DataFrame(lists)

    # 指定保存的CSV文件名
    csv_file = 'smoreg.csv'

    # 使用Pandas保存为CSV文件
    df.to_csv(csv_file, index=False)

    print("CSV文件已保存:", csv_file)
getmodel(admin,all_medians)

def gettdata(all_medians):
    lists = []
    for i in all_medians:
        k = 0
        
        list2 = []
        name1 = ''
        name2 = ''
        for j in all_medians:
           list1 = []
           str_t = i[0].split('-')[0]
           str_s = j[0].split('-')[0]
           if str_t != str_s:
               list1.append(j[0])
               list1.append(i[0])
               list1+= j[1:]
               list1+= i[1:]
               list2.append(list1)




            # 转换为Pandas DataFrame
        df = pd.DataFrame(list2)

        # 指定保存的CSV文件名
        csv_file = '../../testdata5/'+i[0]+'_test.csv'

        # 使用Pandas保存为CSV文件
        df.to_csv(csv_file, index=False)

        print("CSV文件已保存:", csv_file)
gettdata(all_medians)