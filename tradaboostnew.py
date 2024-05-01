# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/16 10:47
@Function:
"""
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class TrAdaboost:
    def __init__(self, base_classifier=DecisionTreeClassifier(), N=10):
        self.base_classifier = base_classifier
        self.N = N
        self.beta_all = np.zeros([1, self.N])
        self.classifiers = []

    def fit(self, x_source, x_target, y_source, y_target):
        x_train = np.concatenate((x_source, x_target), axis=0)
        y_train = np.concatenate((y_source, y_target), axis=0)
        x_train = np.asarray(x_train, order='C')
        y_train = np.asarray(y_train, order='C')
        y_source = np.asarray(y_source, order='C')
        y_target = np.asarray(y_target, order='C')

        row_source = x_source.shape[0]
        row_target = x_target.shape[0]

        # 初始化权重
        weight_source = np.ones([row_source, 1]) / row_source
        weight_target = np.ones([row_target, 1]) / row_target
        weights = np.concatenate((weight_source, weight_target), axis=0)

        beta = 1 / (1 + np.sqrt(2 * np.log(row_source / self.N)))

        result = np.ones([row_source + row_target, self.N])
        for i in range(self.N):
            weights = self._calculate_weight(weights)
            self.base_classifier.fit(x_train, y_train, sample_weight=weights[:, 0])
            self.classifiers.append(self.base_classifier)

            result[:, i] = self.base_classifier.predict(x_train)
            error_rate = self._calculate_error_rate(y_target,
                                                    result[row_source:, i],
                                                    weights[row_source:, :])

            print("Error Rate in target data: ", error_rate, 'round:', i, 'all_round:', self.N)

            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                self.N = i
                print("Early stopping...")
                break
            self.beta_all[0, i] = error_rate / (1 - error_rate)

            # 调整 target 样本权重 正确样本权重变大
            for t in range(row_target):
                weights[row_source + t] = weights[row_source + t] * np.power(self.beta_all[0, i], -np.abs(result[row_source + t, i] - y_target[t]))
            # 调整 source 样本 错分样本变大
            for s in range(row_source):
                weights[s] = weights[s] * np.power(beta, np.abs(result[s, i] - y_source[s]))

    def predict(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))

            if left >= right:
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def predict_prob(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))
            predict.append([left, right])
        return predict

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, y_target, y_predict, weight_target):
        sum_weight = np.sum(weight_target)
        return np.sum(weight_target[:, 0] / sum_weight * np.abs(y_target - y_predict))
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def main():
    file_path = 'multitrada.csv'  # CSV文件路径
    data = pd.read_csv(file_path)
    f1list = []
    # 获取第一列和第二列的内容
    list1 = data.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    print(list1)

    list2 = data.iloc[:, 1].tolist()  # 获取第二列并转换为列表

    list3 = []
    for file1, file2 in zip(list1, list2):

        file_path1 = f'Tradaboostdata/{file2}'
        file_path2 = f'bugdata3/{file2}'
        # 读取源项目的CSV文件
        source_data = pd.read_csv(file_path1)
        x_source = source_data.iloc[:, 1:-1]  # 假设最后一列是标签
        y_source = source_data.iloc[:, -1]

        # 读取目标项目的CSV文件
        target_data = pd.read_csv(file_path2)
        x_target = target_data.iloc[:, 1:-1]  # 假设最后一列是标签
        y_target = target_data.iloc[:, -1]
        y_source = y_source.apply(lambda x: 1 if x > 0 else 0)
        y_target = y_target.apply(lambda x: 1 if x > 0 else 0)
        # 初始化 Logistic Regression 分类器
        lr_classifier = LogisticRegression(max_iter=1000)


        # 初始化 TrAdaboost 算法
        tradaboost = TrAdaboost(base_classifier=lr_classifier, N=10)

        # 训练 TrAdaboost 算法
        tradaboost.fit(x_source, x_target, y_source, y_target)

        # 在目标项目上进行预测
        predictions = tradaboost.predict(x_target)
        print(predictions)
        # 计算预测的f1值
        f1 = f1_score(y_target, predictions)
        f1list.append(f1)
    for i in f1list:
        print(i)
if __name__ == "__main__":
    main()
