# coding: UTF-8
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.metrics import f1_score

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
    tarin_data = pd.read_csv('TCAALL/camel-1.0_merged.csv')
    # test data
    test_data = pd.read_csv('bugdata3/camel-1.0.csv')

    trans_S = tarin_data.iloc[:, 1:-1]
    label_S = tarin_data.iloc[:, -1]
    label_S = label_S.apply(lambda x: 1 if x > 0 else 0)
    test = test_data.iloc[:, 1:-1]
    label_test = test_data.iloc[:, -1]
    label_test = label_test.apply(lambda x: 1 if x > 0 else 0)
    N = 10

    predict, k, n = TrAdaBoost(trans_S, label_S, test, N)

    f1 = f1_score(label_test, predict)
    print("F1 Score:", f1)
