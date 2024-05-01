"""
Kernel Mean Matching
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
"""
import pandas as pd
import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--norm', action='store_true')
args = parser.parse_args()

def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K

class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta

def load_data(folder, domain):
    from scipy import io
    data = io.loadmat(os.path.join(folder, domain + '_fc6.mat'))
    return data['fts'], data['labels']


def knn_classify(Xs, Ys, Xt, Yt, k=1, norm=False):
    # model = KNeighborsClassifier(n_neighbors=k)
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    Ys = Ys.ravel()

    Yt = Yt.ravel()
    if norm:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs)
        Xt = scaler.fit_transform(Xt)
    model.fit(Xs, Ys)
    Yt_pred = model.predict(Xt)
    acc = accuracy_score(Yt, Yt_pred)
    print(f'Accuracy using kNN: {acc * 100:.2f}%')
    f1 = f1_score(Yt, Yt_pred, average='weighted')
    auc = roc_auc_score(Yt, Yt_pred, average='weighted')
    return f1

if __name__ == "__main__":
    # download the dataset here: https://www.jianguoyun.com/p/DcNAUg0QmN7PCBiF9asD (Password: qqLA7D)
    file_path = 'output_f1_spe+.csv'  # CSV文件路径
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

        kmm = KMM(kernel_type='rbf', B=10)
        beta = kmm.fit(Xs, Xt)
        print(beta)
        print(beta.shape)
        Xs_new = beta * Xs
        f1=knn_classify(Xs_new, Ys, Xt, Yt, k=1, norm=args.norm)
        list3.append(f1)
    for i in list3:
        print(i)