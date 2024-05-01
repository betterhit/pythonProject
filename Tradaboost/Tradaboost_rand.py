# coding: UTF-8
import os
import random

import numpy as np
import copy
from sklearn import tree
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


# =============================================================================
# Public estimators
# =============================================================================

def TrAdaBoost(trans_S, trans_A, label_S, label_A, test, N=20):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    if N > row_A:
        print('The maximum of iterations should be smaller than ', row_A)

    test_data = np.concatenate((trans_data, test), axis=0)

    # Initialize the weights
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # Save prediction labels and bata_t
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    # Save the prediction labels of test data
    predict = np.zeros([row_T])
    print('params initial finished.')
    print('=' * 60)

    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    error_rate_list = []
    misclassify_list = []
    for i in range(N):
        P = calculate_P(weights)
        result_label[:, i] = train_classify(trans_data, trans_label, test_data, P)
        error_rate, misclassify = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                                       weights[row_A:row_A + row_S, :])
        if error_rate > 0.5:
            error_rate = 1 - error_rate
            # for a binary classifier
            # reverse the prediction label 0 to 1; 1 to 0.
            pre_labels = copy.deepcopy(result_label[:, i])
            result_label[:, i] = np.invert(pre_labels.astype(np.int32)) + 2
        # Avoiding overfitting
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
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_A + j, i] - label_S[j])))
        # Changing the data weights of diff-distribution training data
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))

    for i in range(row_T):
        left = np.sum(
            result_label[row_A + row_S + i, int(np.floor(N / 2)):N] * np.log(1 / bata_T[0, int(np.floor(N / 2)):N]))
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
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=3, class_weight='balanced', max_features="log2",
                                      splitter="best", random_state=0)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)
    misclassify = np.sum(np.abs(label_R - label_H)) / len(label_H)
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H)), misclassify,

def rand_predict(folder_path):
    list3 = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".csv"):
                # 构建完整的文件路径
                target_path = os.path.join(root, filename)

                root1 = 'C:\\Users\\fuhang\Desktop\\3sw'
                folder_path1 = os.path.join(root1, filename[0:-4])

                csv_files = [file for file in os.listdir(folder_path1) if file.endswith('.csv')]

                # 随机选择一个 CSV 文件
                random_csv_files = random.sample(csv_files, 2)

                # 构建完整的文件路径
                source_paths = [os.path.join(folder_path1, file_name) for file_name in random_csv_files]

                test_data = pd.read_csv(target_path)
                tarin_data = pd.read_csv(source_paths[0])
                A_tarin_data = pd.read_csv(source_paths[1])
                trans_S = tarin_data.iloc[:, 1:-1]
                trans_A = A_tarin_data.iloc[:, 1:-1]
                label_S = tarin_data.iloc[:, -1]
                label_A = A_tarin_data.iloc[:, -1]
                label_S = label_S.apply(lambda x: 1 if x > 0 else 0)
                label_A = label_A.apply(lambda x: 1 if x > 0 else 0)
                test = test_data.iloc[:, 1:-1]
                label_test = test_data.iloc[:, -1]
                label_test = label_test.apply(lambda x: 1 if x > 0 else 0)
                N = 10

                predict, k, n = TrAdaBoost(trans_S, trans_A, label_S, label_A, test, N)
                f1 = f1_score(label_test, predict, average='weighted')
                auc = roc_auc_score(label_test, predict, average='weighted')
                list3.append(auc)
    return  list3

if __name__ == '__main__':
    folder_path = '../bugdata3'
    df = pd.DataFrame()

    # 重复执行 rand_predict 函数并将结果添加到DataFrame的新列中
    for i in range(10):
        result_list = rand_predict(folder_path)  # 获取一维列表
        df[f'Column_{i + 1}'] = result_list  # 将结果列表添加到DataFrame的新列中

    # 将DataFrame保存到CSV文件中
    df.to_csv('Tradaboost_auc.csv', index=False)