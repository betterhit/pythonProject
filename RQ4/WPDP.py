import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
# 读取数据
folder_path = '../bugdata3'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
for i in range(len(csv_files)):
            df1 = pd.read_csv(os.path.join(folder_path, csv_files[i]))
            Xs = df1.iloc[:, 1:20].values
            Ys = df1.iloc[:, 21].values
            Ys[Ys != 0] = 1  # 将非零值标记为1

            # 定义模型
            logistic_model = LogisticRegression(solver='liblinear')

            # 设置随机种子


            # 执行50次预测
            f1_scores = []
            for _ in range(50):
                # 随机划分数据集
                X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.5, stratify=Ys)

                # 训练模型
                logistic_model.fit(X_train, y_train)

                # 预测
                y_pred = logistic_model.predict(X_test)

                # 计算F1分数
                f1 = f1_score(y_test, y_pred,average='weighted')

                # 保存F1分数
                f1_scores.append(f1)

            # 计算平均F1分数
            average_f1_score = np.mean(f1_scores)
            print(average_f1_score)
