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
    return auc

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
    folder_path = '../../bugdata3'
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

                kmm = KMM(kernel_type='rbf', B=10)
                beta = kmm.fit(Xs, Xt)
                print(beta)
                print(beta.shape)
                Xs_new = beta * Xs
                f1 = knn_classify(Xs_new, Ys, Xt, Yt, k=1, norm=args.norm)
                result = [csv_files[i], csv_files[j], f1]
                results.append(result)

    # 打印所有结果
    for result in results:
        print(result)
    with open('adm_auc.txt', 'w') as f:
        for i in results:
            for j in i:
                f.write(str(j))

                f.write(' ')
            f.write('\n')
        f.close()
