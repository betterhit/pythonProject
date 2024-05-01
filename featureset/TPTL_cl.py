
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
    def __init__(self, kernel_type='primal', dim=12, lamb=1, gamma=1):
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
    def fit(self,Xs,Xt,X):
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        H = np.eye(n) - 1 / n * np.ones((n, n))

        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n

        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])

        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt, X):

        Xs_new, Xt_new = self.fit(Xs, Xt, X)

        feature_columns = [f'Feature{i + 1}' for i in range(Xs_new.shape[1])]
        data = np.column_stack((Xs_new, Ys))  # 将特征数据和标签数据合并为一个数组
        df = pd.DataFrame(data, columns=feature_columns + ['Label'])
        feature_columns2 = [f'Feature{i + 1}' for i in range(Xt_new.shape[1])]
        data2 = np.column_stack((Xt_new, Yt))  # 将特征数据和标签数据合并为一个数组
        df2 = pd.DataFrame(data2, columns=feature_columns2 + ['Label'])


        clf = LogisticRegression(solver='liblinear')
        # label_mapping = {"buggy": 1, "clean": 0}

        # 使用字典将文本标签映射为整数

        # Yt_encoded和y_pred_encoded现在包含整数标签，可以用于计算F1值
        # 在源域数据上训练逻辑回归分类器
        clf.fit(Xs_new, Ys)

        # 使用分类器在目标域数据上进行预测
        y_pred = clf.predict(Xt_new)
        # Yt_encoded = [label_mapping[label] for label in Yt]  # 假设Yt是文本标签的数组

        # y_pred_encoded = [label_mapping[label] for label in y_pred]  # 假设y_pred是文本标签的数组

        # 计算 F1 得分
        f1 = f1_score(Yt, y_pred,average='weighted')
        # f1 = f1_score(Yt, y_pred)
        return f1, y_pred




if __name__ == '__main__':
    file_path = 'sort_proj.csv'  # CSV文件路径
    data = pd.read_csv(file_path)
    file_columns = []

    with open('cl.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(',')
            file_name = elements[0]
            columns = list(map(int, elements[1:]))
            file_columns.append([file_name, columns])
    print(file_columns)

    # 获取第一列和第二列的内容
    list1 = data.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    list2 = data.iloc[:, 1].tolist()  # 获取第二列并转换为列表
    print(list1)
    print(list2)
    list3=[]
    for file1, file2 in zip(list1, list2):
        for item in file_columns:
            file_name = item[0]
            columns_to_select = item[1][0:11]
            if file2 == file_name:
                file_path1 = f'../bugdata3/{file1}'
                file_path2 = f'../bugdata3/{file2}'
                df1 = pd.read_csv(file_path1)
                df2 = pd.read_csv(file_path2)

                # 选择特定列
                Xs = df2.iloc[:, columns_to_select].values
                Ys = df2.iloc[:, 21].values
                Ys[Ys != 0] = 1

                Xt = df1.iloc[:, columns_to_select].values
                Yt = df1.iloc[:, 21].values
                Yt[Yt != 0] = 1
                Xs = np.array(Xs)
                Ys = np.array(Ys)
                Xt = np.array(Xt)
                Yt = np.array(Yt)
                tca = TCA(kernel_type='linear',lamb=1, gamma=1)
                sort = 'N2'
                X = tca.tca_plus(Xs, Xt)
                acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt, Yt, X)
                list3.append(acc1)
    for i in list3:
        print(i)