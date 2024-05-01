# encoding=utf-8
"""
    Created on 21:29 2018/11/12
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

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
        print(C)
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(0, C):
                    print(c)
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    # print(inds)
                    # print(Y_tar_pseudo)
                    # print(len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]))
                    e[tuple(inds)] = -1 / max(len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]),1)
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            f1 = f1_score(Yt, Y_tar_pseudo, average='weighted')
            auc = roc_auc_score(Yt, Y_tar_pseudo, average='weighted')
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return f1, Y_tar_pseudo, list_acc


# if __name__ == '__main__':
#     domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
#     for i in range(1):
#         for j in range(2):
#             if i != j:
#                 src, tar = 'data/' + domains[i], 'data/' + domains[j]
#                 src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
#                 Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
#                 jda = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
#                 acc, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)
#                 print(acc)
if __name__ == '__main__':
    # file_path = '../singlesource/MZ_Score/selected_projects.csv'
    # file_path = '../singlesource/Bellwether/bellwether.csv'  # CSV文件路径
    # file_path = '../singlesource/CFPS/CFPS_AUC.csv'
    file_path = 'output_f1_spe+.csv'
    data = pd.read_csv(file_path)

    # 获取第一列和第二列的内容
    list1 = data.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    list2 = data.iloc[:, 1].tolist()  # 获取第二列并转换为列表
    print(list1)
    print(list2)
    list3 = []
    for file1, file2 in zip(list1, list2):
        file_path1 = f'../bugdata3/{file1}'
        file_path2 = f'../bugdata3/{file2}'
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
        Xs = df2.iloc[:, 1:20].values  # 选择第1列到第21列作为 xs
        Ys = df2.iloc[:, 21].values  # 选择第22列作为 ys
        Ys[Ys != 0] = 1
        Xt = df1.iloc[:, 1:20].values  # 选择第1列到第21列作为 xs
        Yt = df1.iloc[:, 21].values  # 选择第22列作为 ys
        Yt[Yt != 0] = 1
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        Xt = np.array(Xt)
        Yt = np.array(Yt)
        jda = JDA(kernel_type='primal', dim=10, lamb=1, gamma=1)
        f1, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)

        list3.append(f1)
    for i in list3:
        print(i)