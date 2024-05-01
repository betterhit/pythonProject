# encoding=utf-8
"""
    Created on 21:29 2018/11/12
    @author: Jindong Wang
"""
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=10, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        print(len(Xs))
        print(len(Xt))
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        feature_columns = [f'Feature{i + 1}' for i in range(Xs_new.shape[1])]
        data = np.column_stack((Xs_new, Ys))  # 将特征数据和标签数据合并为一个数组
        df = pd.DataFrame(data, columns=feature_columns + ['Label'])

        # 保存 DataFrame 到 CSV 文件
        df.to_csv('output/Safe_non.csv', index=False)
        feature_columns2 = [f'Feature{i + 1}' for i in range(Xt_new.shape[1])]
        data2 = np.column_stack((Xt_new, Yt))  # 将特征数据和标签数据合并为一个数组
        df2 = pd.DataFrame(data2, columns=feature_columns2 + ['Label'])

        # 保存 DataFrame 到 CSV 文件
        df2.to_csv('output/Apache_non.csv', index=False)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)

        return acc, y_pred

    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]

        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1=Xt2, X2=X, gamma=self.gamma)

        # New target features
        Xt2_new = K @ A

        return Xt2_new

    def fit_predict_new(self, Xt, Xs, Ys, Xt2, Yt2):
        '''
        Transfrom Xt and Xs, get Xs_new
        Transform Xt2 with projection matrix created by Xs and Xt, get Xt2_new
        Make predictions on Xt2_new using classifier trained on Xs_new
        :param Xt: ns * n_feature, target feature
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt2: nt * n_feature, new target feature
        :param Yt2: nt * 1, new target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, _ = self.fit(Xs, Xt)
        Xt2_new = self.fit_new(Xs, Xt, Xt2)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)

        return acc, y_pred


if __name__ == '__main__':
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    # for i in [1]:
    #     for j in [2]:
    #         if i != j:
    #             src, tar = 'data/decaf/' + domains[i], 'data/decaf/' + domains[j]
    #             src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    #            #print('src_domain type:', type(src_domain), '\nsrc_domain:', src_domain)
    #             #print(src_domain)
    #             Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['labels'], tar_domain['feas'], tar_domain['labels']
    #             print(Ys)
    #             # Split target data
    #             Xt1, Xt2, Yt1, Yt2 = train_test_split(Xt, Yt, train_size=50, stratify=Yt, random_state=42)
    #
    #             # Create latent space and evaluate using Xs and Xt1
    #             tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    #             acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt1, Yt1)
    #
    #             # Project and evaluate Xt2 existing projection matrix and classifier
    #             acc2, ypre2 = tca.fit_predict_new(Xt1, Xs, Ys, Xt2, Yt2)
    df1 = pd.read_csv('apache.csv')
    df2 = pd.read_csv('safe.csv')

    # 选择需要的列作为 xs 和 ys
    Xs = df2.iloc[:, 0:25].values  # 选择第1列到第21列作为 xs
    Ys = df2.iloc[:, 26].values  # 选择第22列作为 ys
    Xt = df1.iloc[:, 0:25].values  # 选择第1列到第21列作为 xs
    Yt = df1.iloc[:, 26].values  # 选择第22列作为 ys
    # print(Xs)
    # print(type(Ys[0]))
    # 将 xs 和 ys 转换为 NumPy 数组
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Xt = np.array(Xt)
    Yt = np.array(Yt)

    Xt1, Xt2, Yt1, Yt2 = train_test_split(Xt, Yt, train_size=50, stratify=Yt, random_state=42)

    tca = TCA(kernel_type='linear', dim=20, lamb=1, gamma=1)

    acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt1, Yt1)
    print(Xs[1])
    print(Xt[1])

                        # Project and evaluate Xt2 existing projection matrix and classifier
    acc2, ypre2 = tca.fit_predict_new(Xt1, Xs, Ys, Xt2, Yt2)
    print(f'Accuracy of mapped source and target1 data : {acc1:.3f}')  # 0.800
    print(f'Accuracy of mapped target2 data            : {acc2:.3f}')  # 0.706