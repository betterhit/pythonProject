# encoding=utf-8
"""
    Created on 9:52 2018/11/14
    @author: Jindong Wang
"""
import os

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn import metrics
from sklearn import svm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int),
                         np.ones(nb_target, dtype=int)))

    clf = svm.LinearSVC(random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


def estimate_mu(_X1, _Y1, _X2, _Y2):
    adist_m = proxy_a_distance(_X1, _X2)
    C = len(np.unique(_Y1))
    epsilon = 1e-3
    list_adist_c = []
    for i in range(1, C + 1):
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    adist_c = sum(list_adist_c) / C
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0
    return mu


class BDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu=False):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode
        self.estimate_mu = estimate_mu

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        print("c:",C)
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = 0
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                labels = [0, 1]  # 假设类别标签是0和1
                for c in labels:
                    e = np.zeros((n, 1))
                    Ns = len(Ys[np.where(Ys == c)])
                    Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    if self.mode == 'WBDA':
                        Ps = Ns / len(Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                        alpha = Pt / Ps
                        mu = 1
                    else:
                        alpha = 1
                    # 更新e矩阵
                    e[np.where(Ys == c)[0]] = 1 / Ns
                    e[np.where(Y_tar_pseudo == c)[0] + ns] = -alpha / Nt
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            # In BDA, mu can be set or automatically estimated using A-distance
            # In WBDA, we find that setting mu=1 is enough
            if self.estimate_mu and self.mode == 'BDA':
                if Xs_new is not None:
                    mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                else:
                    mu = 0
            M = (1 - mu) * M0 + mu * N
            M /= np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot(
                [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            clf = LogisticRegression(solver='liblinear',class_weight='balanced')
            # clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            f1 = f1_score(Yt, Y_tar_pseudo, average='weighted')
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            auc = roc_auc_score(Yt, Y_tar_pseudo, average='weighted')
            list_acc.append(acc)
            print('{} iteration [{}/{}]: Acc: {:.4f}'.format(self.mode, t + 1, self.T, acc))
        return auc, Y_tar_pseudo, list_acc


if __name__ == '__main__':

    # df1 = pd.read_csv('bugdata/ant/ant-1.3.csv')
    # df2 = pd.read_csv('bugdata2/camel/camel-1.6.csv')
    # Xs = df2.iloc[:, 1:20].values  # 选择第1列到第21列作为 xs
    # Ys = df2.iloc[:, 21].values  # 选择第22列作为 ys
    # Ys[Ys != 0] = 1
    # Xt = df1.iloc[:, 1:20].values  # 选择第1列到第21列作为 xs
    # Yt = df1.iloc[:, 21].values  # 选择第22列作为 ys
    # Yt[Yt != 0] = 1
    # Xs = np.array(Xs)
    # Ys = np.array(Ys)
    # Xt = np.array(Xt)
    # Yt = np.array(Yt)
    # tca = TCA(kernel_type='linear', dim=15, lamb=1, gamma=1)
    # # print(Xs)
    # sort = 'N2'
    # # X = tca.select_norm(Xs,Xt,sort)
    # X = tca.tca_plus(Xs, Xt)
    # acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt, Yt,X)
    # print(f'Accuracy of mapped source and target1 data : {acc1:.3f}')  # 0.800
    folder_path = '../bugdata3'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    results = []  # 用于存储每一对组合的结果，可以使用字典存储
    result_key = []
    # for i in range(len(csv_files)):
    #     for j in range(i + 1, len(csv_files)):  # 遍历所有文件的组合
    #         df2 = pd.read_csv(os.path.join(folder_path, csv_files[i]))
    #         df1 = pd.read_csv(os.path.join(folder_path, csv_files[j]))
    #         print(csv_files[i])
    #         print(csv_files[j])
    #         Xs = df2.iloc[:, 1:20].values
    #         Ys = df2.iloc[:, 21].values
    #         Ys[Ys != 0] = 1
    #
    #         Xt = df1.iloc[:, 1:20].values
    #         Yt = df1.iloc[:, 21].values
    #         Yt[Yt != 0] = 1
    #
    #         Xs = np.array(Xs)
    #         Ys = np.array(Ys)
    #         Xt = np.array(Xt)
    #         Yt = np.array(Yt)
    #
    #         tca = TCA(kernel_type='linear', dim=15, lamb=1, gamma=1)
    #         sort = 'N2'
    #         X = tca.tca_plus(Xs, Xt)
    #         acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt, X)
    #         result = [csv_files[i], csv_files[j], acc]
    #         results.append(result)
    for i in range(len(csv_files)):
        for j in range(len(csv_files)):  # 遍历所有文件的组合，包括相同文件
            str_t = csv_files[i].split('-')[0]
            str_s = csv_files[j].split('-')[0]
            if i != j and str_s!= str_t: # 排除同一个文件的组合
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


                bda = BDA(kernel_type='primal', dim=10, lamb=1, mu=0.5,
                          mode='BDA', gamma=1, estimate_mu=False)
                f1, ypre, list_acc = bda.fit_predict(Xs, Ys, Xt, Yt)
                result = [csv_files[i], csv_files[j], f1]
                results.append(result)

    # 打印所有结果
    for result in results:
        print(result)
    with open('adm_auc_new.txt', 'w') as f:
        for i in results:
            for j in i:
                f.write(str(j))

                f.write(' ')
            f.write('\n')
        f.close()
