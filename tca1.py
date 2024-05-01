import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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


def sort_degree(x, y):
    print("hhhhh")
    print(x,y)
    if x * 1.6 < y:
        return 7;
    elif x * 1.3 < y and y <= x * 1.6:
        return 6;
    elif x * 1.1 < y and y <= x * 1.3:
        return 5;
    elif x * 0.9 < y and y <= x * 1.1:
        return 4;
    elif x * 0.7 < y and y <= x * 0.9:
        return 3;
    elif x * 0.4 < y and y <= x * 0.7:
        return 2;
    elif y <= x * 0.4:
        return 1;
class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma




    def tca_plus(self,Xs,Xt):


        # 获取矩阵的行数
        ns = Xs.shape[0]
        dist1 = []
        # 遍历矩阵中的每两行
        for i in range(ns):
            for j in range(i + 1, ns):  # 注意 j 从 i+1 开始，以避免计算重复的距离
                # 计算欧氏距离
                diff = Xs[i] - Xs[j]
                euclidean_distance = np.linalg.norm(diff)

                # 将距离添加到列表
                dist1.append(euclidean_distance)
        mean1 = np.mean(dist1)
        median1= np.median(dist1)
        min1= np.min(dist1)
        max1= np.max(dist1)
        std1 = np.std(dist1)
        nt = Xt.shape[0]
        print("ns:",ns)
        print("nt:",nt)
        print("sort:",sort_degree(ns,nt))
        dist2 = []
        # 遍历矩阵中的每两行
        for i in range(nt):
            for j in range(i + 1, nt):  # 注意 j 从 i+1 开始，以避免计算重复的距离
                # 计算欧氏距离
                diff = Xt[i] - Xt[j]
                euclidean_distance = np.linalg.norm(diff)
                # 将距离添加到列表
                dist2.append(euclidean_distance)
        mean2 = np.mean(dist2)
        median2 = np.median(dist2)
        min2 = np.min(dist2)
        max2 = np.max(dist2)
        std2 = np.std(dist2)
        if sort_degree(mean1,mean2)==4 and sort_degree(std1,std2)==4:
            X = np.hstack((Xs.T, Xt.T))
            print("N1")
            return X
        elif (sort_degree(min1,min2)==7 or sort_degree(min1,min2)==1) and (sort_degree(max1,max2)==7 or sort_degree(max1,max2)==1) and (sort_degree(ns,nt)==7 or sort_degree(ns,nt)==1):
            X = np.vstack((Xs, Xt))
            X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X = X.T
            print("N2")
            return X
        elif (sort_degree(std1,std2)==7 and sort_degree(ns,nt)<4) or (sort_degree(std1,std2)==1 and sort_degree(ns,nt) > 4):
            Xs = (Xs - mean1) / std1
            Xt = (Xt - mean1) / std1
            X = np.vstack((Xs, Xt))
            X = X.T
            print("N3")
            return X
        elif (sort_degree(std1,std2) == 7 and sort_degree(ns,nt) == 7) or (sort_degree(std1,std2)==7 and sort_degree(ns,nt)==1):
            Xs = (Xs - mean2) / std2
            Xt = (Xt - mean2) / std2
            X = np.vstack((Xs, Xt))
            X = X.T
            print("N4")
            return X
        else :
            X = np.vstack((Xs, Xt))
            mean_values = np.mean(X, axis=0)
            std_values = np.std(X, axis=0)
            X = (X - mean_values) / std_values
            X = X.T
            print("N5")
            return X


    def select_norm(self,Xs,Xt,sort):
        print(type(sort))
        if sort == 'NON':
            X = np.hstack((Xs.T, Xt.T))
            return X
        elif sort == 'N1':
            X = np.vstack((Xs, Xt))
            X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X = X.T
            return X
        elif sort == 'N2':
            X = np.vstack((Xs, Xt))
            mean_values = np.mean(X, axis=0)
            std_values = np.std(X, axis=0)
            X = (X - mean_values) / std_values
            X = X.T
            return X
        elif sort == 'N3':
            mean_values1 = np.mean(Xs, axis=0)
            std_values1 = np.std(Xs, axis=0)
            Xs = (Xs - mean_values1) / std_values1
            Xt = (Xt - mean_values1) / std_values1
            X = np.vstack((Xs, Xt))
            X = X.T
            return X
        elif sort == 'N4':
            mean_values2 = np.mean(Xt, axis=0)
            std_values2 = np.std(Xt, axis=0)
            Xs = (Xs - mean_values2) / std_values2
            Xt = (Xt - mean_values2) / std_values2
            X = np.vstack((Xs, Xt))
            X = X.T
            return X
        elif sort == 'N5':
            X = np.hstack((Xs.T, Xt.T))
            X /= np.linalg.norm(X, axis=0)
            return X
        else:
            return -1
    # def select_norm2(self,Xs,Xt,sort):
    #     print(type(sort))
    #     if sort == 'NON':
    #         X = np.hstack((Xs.T, Xt.T))
    #         return X
    #     elif sort == 'N1':
    #         X = np.hstack((Xs.T, Xt.T))
    #         X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    #
    #         return X
    #     elif sort == 'N2':
    #         X = np.hstack((Xs.T, Xt.T))
    #         mean_values = np.mean(X, axis=0)
    #         std_values = np.std(X, axis=0)
    #         X = (X - mean_values) / std_values
    #
    #         return X
    #     elif sort == 'N3':
    #         mean_values1 = np.mean(Xs.T, axis=0)
    #         std_values1 = np.std(Xs.T, axis=0)
    #         Xs = (Xs - mean_values1) / std_values1
    #         Xt = (Xt - mean_values1) / std_values1
    #         X = np.vstack((Xs, Xt))
    #         X = X.T
    #         return X
    #     elif sort == 'N4':
    #         mean_values2 = np.mean(Xt, axis=0)
    #         std_values2 = np.std(Xt, axis=0)
    #         Xs = (Xs - mean_values2) / std_values2
    #         Xt = (Xt - mean_values2) / std_values2
    #         X = np.vstack((Xs, Xt))
    #         X = X.T
    #         return X
    #     elif sort == 'N5':
    #         X = np.hstack((Xs.T, Xt.T))
    #         X /= np.linalg.norm(X, axis=0)
    #         return X
    #     else:
    #         return -1
    def fit(self,Xs,Xt,X):
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        # M = M / np.linalg.norm(M, 'fro')
        # M = (M - M.min()) / (M.max() - M.min())

        H = np.eye(n) - 1 / n * np.ones((n, n))
        print(self.kernel_type)
        print(self.gamma)
        print(X.shape)
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        print(K.shape)
        print(M.shape)
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])

        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        # Z /= np.linalg.norm(Z, axis=0)
        #Z = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0))

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt,X):
        print(X.shape)
        Xs_new, Xt_new = self.fit(Xs,Xt,X)

        feature_columns = [f'Feature{i + 1}' for i in range(Xs_new.shape[1])]
        data = np.column_stack((Xs_new, Ys))  # 将特征数据和标签数据合并为一个数组
        df = pd.DataFrame(data, columns=feature_columns + ['Label'])
        # df.to_csv('output/Safe_non.csv', index=False)
        df.to_csv('output/Apache_non.csv', index=False)

        feature_columns2 = [f'Feature{i + 1}' for i in range(Xt_new.shape[1])]
        data2 = np.column_stack((Xt_new, Yt))  # 将特征数据和标签数据合并为一个数组
        df2 = pd.DataFrame(data2, columns=feature_columns2 + ['Label'])
        # df2.to_csv('output/Apache_non.csv', index=False)
        # df2.to_csv('output/Safe_non.csv', index=False)
        df2.to_csv('output/Zxing_non.csv', index=False)
        # clf = KNeighborsClassifier(n_neighbors=1)
        # clf.fit(Xs_new, Ys.ravel())
        # y_pred = clf.predict(Xt_new)
        # acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        # return acc, y_pred
        clf = LogisticRegression(solver='liblinear')
        # label_mapping = {"buggy": 1, "clean": 0}

        # 使用字典将文本标签映射为整数


        # Yt_encoded和y_pred_encoded现在包含整数标签，可以用于计算F1值
        # 在源域数据上训练逻辑回归分类器
        clf.fit(Xs_new, Ys)

        # 使用分类器在目标域数据上进行预测
        y_pred = clf.predict(Xt_new)
        # Yt_encoded = [label_mapping[label] for label in Yt]  # 假设Yt是文本标签的数组
        print(Yt)
        # y_pred_encoded = [label_mapping[label] for label in y_pred]  # 假设y_pred是文本标签的数组
        print(y_pred)
        # 计算 F1 得分
        f1 = f1_score(Yt, y_pred,average='weighted')

        return f1, y_pred


if __name__ == '__main__':
    # df1 = pd.read_csv('zxing.csv')
    # df2 = pd.read_csv('apache.csv')
    # df1 = pd.read_csv('safe.csv')
    # df2 = pd.read_csv('apache.csv')
    # df1 = pd.read_csv('apache.csv')
    # df2 = pd.read_csv('safe.csv')
    df1 = pd.read_csv('bugdata3/ant-1.3.csv')
    df2 = pd.read_csv('TCAALL/ant-1.3_merged.csv')
    Xs = df2.iloc[:, 1:20].values  # 选择第1列到第21列作为 xs
    Ys = df2.iloc[:, 21].values  # 选择第22列作为 ys
    Ys[Ys != 0] = 1
    Xt = df1.iloc[:, 1:20].values  # 选择第1列到第21列作为 xs
    Yt = df1.iloc[:, 21].values  # 选择第22列作为 ys
    Yt[Yt != 0] = 1
    # Xs = df2.iloc[:, 0:25].values  # 选择第1列到第21列作为 xs
    # Ys = df2.iloc[:, 26].values  # 选择第22列作为 ys
    # Xt = df1.iloc[:, 0:25].values  # 选择第1列到第21列作为 xs
    # Yt = df1.iloc[:, 26].values  # 选择第22列作为 ys
    Xs = np.array(Xs)


    Ys = np.array(Ys)
    Xt = np.array(Xt)
    Yt = np.array(Yt)
    # Xt1, Xt2, Yt1, Yt2 = train_test_split(Xt, Yt, train_size=50, stratify=Yt, random_state=42)
    tca = TCA(kernel_type='linear', dim=15, lamb=1, gamma=1)
    # print(Xs)
    sort = 'N2'
    # X = tca.select_norm(Xs,Xt,sort)
    X = tca.tca_plus(Xs, Xt)
    acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt, Yt,X)
    # acc2, ypre2 = tca.fit_predict_new(Xt1, Xs, Ys, Xt2, Yt2)
    print(f'Accuracy of mapped source and target1 data : {acc1:.3f}')  # 0.800
    # print(f'Accuracy of mapped target2 data            : {acc2:.3f}')  # 0.70