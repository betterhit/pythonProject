import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
# 训练模型


# 从训练集CSV文件中加载数据
train_data = pd.read_csv("smoreg.csv")



# 拆分特征和目标
X_train = train_data.iloc[:, 2:41]



y_train = train_data.iloc[:, 42]

# svr = SVR(kernel='poly')
svr = SVR(
    kernel='poly'
)
# 建立SVR模型
# svr = LinearRegression()

# 训练模型
svr.fit(X_train, y_train)

# 保存模型到文件
joblib.dump(svr, 'svr_model.pkl')


