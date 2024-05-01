# coding: UTF-8
import os

import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def TrAdaBoost(trans_S, label_S, test, N=20):
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_S, test), axis=0)

    # Initialize the weights
    weights_S = np.ones([row_S, 1]) / row_S

    # Save prediction labels and bata_t
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_S + row_T, N])

    # Save the prediction labels of test data
    predict = np.zeros([row_T])
    print('params initial finished.')
    print('=' * 60)

    trans_data = np.asarray(trans_S, order='C')
    test_data = np.asarray(test_data, order='C')

    error_rate_list = []
    misclassify_list = []
    for i in range(N):
        P = calculate_P(weights_S)
        result_label[:, i] = train_classify(trans_data, label_S, test_data, P)
        error_rate, misclassify = calculate_error_rate(label_S, result_label[:row_S, i], weights_S)
        if error_rate > 0.5:
            error_rate = 1 - error_rate
            pre_labels = np.invert(result_label[:, i].astype(np.int32)) + 2
            result_label[:, i] = pre_labels
        elif error_rate <= 1e-10:
            N = i
            break
        error_rate_list.append(error_rate)
        misclassify_list.append(misclassify)
        bata_T[0, i] = error_rate / (1 - error_rate)
        print('Iter {}-th result :'.format(i))
        print('error rate :', error_rate, '|| bata_T :', error_rate / (1 - error_rate))
        print('-' * 60)
        # Changing the data weights of same-distribution training data
        for j in range(row_S):
            weights_S[j] = weights_S[j] * np.power(bata_T[0, i], (-np.abs(result_label[j, i] - label_S[j])))

    for i in range(row_T):
        left = np.sum(result_label[row_S + i, int(np.floor(N / 2)):N] * np.log(1 / bata_T[0, int(np.floor(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.floor(N / 2)):N]))


        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
    print("TrAdaBoost is done")
    print('=' * 60)
    print('The prediction labels of test data are :')
    print(predict)
    return predict, np.round(np.array(error_rate_list), 3), np.round(np.array(misclassify_list), 3)

def calculate_P(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')

def train_classify(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=3, class_weight='balanced', max_features="log2", splitter="best", random_state=0)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)

def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)
    misclassify = np.sum(np.abs(label_R - label_H)) / len(label_H)
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H)), misclassify

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

                predict, k, n = TrAdaBoost(Xs, Ys, Xt, N=10)

                # 在目标域上进行预测

                f1 = f1_score(Yt, predict, average='weighted')
                auc = roc_auc_score(Yt, predict, average='weighted')
                result = [csv_files[i], csv_files[j], f1]
                results.append(result)

    # 打印所有结果
    for result in results:
        print(result)
    with open('adm_f1.txt', 'w') as f:
        for i in results:
            for j in i:
                f.write(str(j))

                f.write(' ')
            f.write('\n')
        f.close()
