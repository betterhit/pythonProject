import os

import pandas as pd
import numpy as np

def fit_predict(Xs, Ys, Xt, Yt):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    # 训练逻辑回归模型
    clf = RandomForestClassifier(n_estimators=100,  # 例如，使用100棵决策树
                                     random_state=42)  # 设置随机状态，以便结果可复现
    # clf = LogisticRegression(solver='sag', max_iter=10000)
    clf.fit(Xs, Ys)

    # 使用训练好的模型进行预测
    Yt_pred = clf.predict(Xt)

    # 计算F1分数
    f1 = f1_score(Yt, Yt_pred)
    auc = roc_auc_score(Yt, Yt_pred, average='weighted')

    return auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


# def fit_predict(Xs, Ys, Xt, Yt):
#     # 使用随机森林训练模型
#     clf = RandomForestClassifier(n_estimators=100,  # 例如，使用100棵决策树
#                                  random_state=42)  # 设置随机状态，以便结果可复现
#     clf.fit(Xs, Ys)
#
#     # 使用训练好的模型进行预测
#     Yt_pred = clf.predict(Xt)
#
#     # 计算F1分数
#     f1 = f1_score(Yt, Yt_pred)
#     print(f1)
#     return f1

def get_simiscores(folder_path,target):
    folder_path=folder_path
    folder_path1 = '../bugdata3'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df1 = pd.read_csv(os.path.join(folder_path1, target))
    features = df1.iloc[:, 1:21]
    # 计算均值和方差
    mean_vector = features.mean()
    std_vector = features.std()
    # 将均值和方差组合成一个一元向量
    NP = np.concatenate((mean_vector.values, std_vector.values))
    simiscores=[]
    for j in range(len(csv_files)):

            df2 = pd.read_csv(os.path.join(folder_path, csv_files[j]))

            features = df2.iloc[:, 1:21]
            # 计算均值和方差
            mean_vector = features.mean()
            std_vector = features.std()
            # 将均值和方差组合成一个一元向量
            HP_j = np.concatenate((mean_vector.values, std_vector.values))
            distance = np.sqrt(np.sum((HP_j - NP) ** 2))
            distance=1000/(1+distance)
            simiscores.append(distance)
    return simiscores
def get_recscores(folder_path):
    folder_path = folder_path
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    results = np.zeros((len(csv_files), len(csv_files)))  # 初始化结果数组为0

    for i in range(len(csv_files)):
        for j in range(len(csv_files)):
            if i != j:  # 排除同一个文件的组合
                df1 = pd.read_csv(os.path.join(folder_path, csv_files[i]))
                df2 = pd.read_csv(os.path.join(folder_path, csv_files[j]))

                Xs = df2.iloc[:, 1:20].values
                Ys = df2.iloc[:, 21].values
                Ys[Ys != 0] = 1

                Xt = df1.iloc[:, 1:20].values
                Yt = df1.iloc[:, 21].values
                Yt[Yt != 0] = 1

                Xs = np.array(Xs)
                Ys = np.array(Ys)
                Xt = np.array(Xt)
                Yt = np.array(Yt)

                f1_score = fit_predict(Xs, Ys, Xt, Yt)
                results[i][j] = f1_score
    np.savetxt('results.csv', results, delimiter=',', fmt='%f')
    return results

if __name__ == '__main__':
    #result_matrix = main()
    folder_path = '../bugdata3'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    #results = np.zeros((len(csv_files), len(csv_files)))  # 初始化结果数组为0
    for i in range(len(csv_files)):
        print(csv_files[i])
        folder_path='C:\\Users\\fuhang\Desktop\\3sw'+'\\'+csv_files[i][0:-4]
        csv_files1 = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        #print(folder_path)
        #results = np.loadtxt('results.csv', delimiter=',', dtype=float)
        simiscores=get_simiscores(folder_path,csv_files[i])
        appscores = get_recscores(folder_path)
        #appscores = results
        recscores = [0] * len(csv_files1)
        for i in range(0,len(simiscores)):
            for j in range(0,len(simiscores)):
                if i != j:
                    recscores[i]+= simiscores[j]*(appscores[j][i]+0.1)
        sort = [index for index, value in sorted(enumerate(recscores), key=lambda x: x[1], reverse=True)]
        with open('result_auc.txt', 'a') as file:
            file.write(csv_files1[sort[1]] + '\n')
        for i in range(0,6):
            print(csv_files1[sort[i]])



