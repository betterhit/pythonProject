import os

import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class TrAdaBoost:
    def __init__(self, iters=10):
        '''
        Implement TrAdaBoost
        To read more about the TrAdaBoost, check the following paper:
            Dai, Wenyuan, et al. "Boosting for transfer learning." Proceedings of the 24th international conference on Machine learning. ACM, 2007.
        The code is modified according to https://github.com/chenchiwei/tradaboost/blob/master/TrAdaboost.py
        '''
        self.iters = iters
    def fit_predict(self, trans_S, trans_A, label_S, label_A, test):
        N = self.iters
        trans_data = np.concatenate((trans_A, trans_S), axis=0)
        trans_label = np.concatenate((label_A, label_S), axis=0)

        row_A = trans_A.shape[0]
        row_S = trans_S.shape[0]
        row_T = test.shape[0]

        test_data = np.concatenate((trans_data, test), axis=0)

        # 初始化权重
        weights_A = np.ones([row_A, 1]) / row_A
        weights_S = np.ones([row_S, 1]) / row_S
        weights = np.concatenate((weights_A, weights_S), axis=0)

        bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

        # 存储每次迭代的标签和bata值？
        bata_T = np.zeros([1, N])
        result_label = np.ones([row_A + row_S + row_T, N])

        predict = np.zeros([row_T])

        # print ('params initial finished.')
        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')
        test_data = np.asarray(test_data, order='C')

        for i in range(N):
            P = self.calculate_P(weights, trans_label)

            result_label[:, i] = self.train_classify(trans_data, trans_label,
                                                test_data, P)
            # print ('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

            error_rate = self.calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                            weights[row_A:row_A + row_S, :])
            print ('Error rate:', error_rate)
            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                N = i
                break  # 防止过拟合
                # error_rate = 0.001

            bata_T[0, i] = error_rate / (1 - error_rate)

            # 调整源域样本权重
            for j in range(row_S):
                weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                                (-np.abs(result_label[row_A + j, i] - label_S[j])))

            # 调整辅域样本权重
            for j in range(row_A):
                weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
        # print bata_T
        for i in range(row_T):
            # 跳过训练数据的标签
            left = np.sum(
                result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
            right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

            if left >= right:
                predict[i] = 1
            else:
                predict[i] = 0
                # print left, right, predict[i]

        return predict


    def calculate_P(self, weights, label):
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')


    def train_classify(self, trans_data, trans_label, test_data, P):
        clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
        clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
        return clf.predict(test_data)


    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)

        # print(weight[:, 0] / total)
        # print(np.abs(label_R - label_H))
        return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))
if __name__ == '__main__':
    folder_path = '../bugdata3'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    results = []  # 用于存储每一对组合的结果，可以使用字典存储
    result_key = []
    for i in range(len(csv_files)):
        for j in range(len(csv_files)):  # 遍历所有文件的组合，包括相同文件
            str_t = csv_files[i].split('-')[0]
            str_s = csv_files[j].split('-')[0]
            if i != j and str_s != str_t:  # 排除同一个文件的组合
                df1 = pd.read_csv(os.path.join(folder_path, csv_files[i]))
                df2 = pd.read_csv(os.path.join(folder_path, csv_files[j]))
                print(csv_files[i])
                print(csv_files[j])
                Xt = df1.iloc[:, 1:20].values
                Yt = df1.iloc[:, 21].values
                Yt[Yt != 0] = 1  # 将目标域的标签转换为二分类标签

                # 划分目标域数据为辅助域和测试集
                X_trans_A, X_test, Y_trans_A, Y_test = train_test_split(Xt, Yt, test_size=0.5, random_state=42)

                # 源域数据
                Xs = df2.iloc[:, 1:20].values
                Ys = df2.iloc[:, 21].values
                Ys[Ys != 0] = 1  # 将源域的标签转换为二分类标签

                # 转换为数组
                Xs = np.array(Xs)
                Ys = np.array(Ys)
                X_trans_A = np.array(X_trans_A)
                Y_trans_A = np.array(Y_trans_A)
                X_test = np.array(X_test)
                Y_test = np.array(Y_test)

                model = TrAdaBoost()
                predict = model.fit_predict(Xs,X_trans_A,Ys,Y_trans_A,X_test)
                # 在目标域上进行预测

                f1 = f1_score(Y_test, predict, average='weighted')
                auc = roc_auc_score(Y_test, predict, average='weighted')
                result = [csv_files[i], csv_files[j], f1]
                results.append(result)

    # 打印所有结果
    for result in results:
        print(result)
    with open('adm_f1_2.txt', 'w') as f:
        for i in results:
            for j in i:
                f.write(str(j))

                f.write(' ')
            f.write('\n')
        f.close()
