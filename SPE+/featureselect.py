import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def feature_shuffle_rf(X_train, y_train, max_depth=None, class_weight=None, top_n=15, n_estimators=50, random_state=0):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, class_weight=class_weight,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = (model.predict_proba(X_train)[:, 1] > 0.5).astype(int)
    train_auc = roc_auc_score(y_train, predictions)
    feature_dict = {}

    # selection  logic
    for feature_index, feature in enumerate(X_train.columns):
        X_train_c = X_train.copy().reset_index(drop=True)
        y_train_c = y_train.copy().reset_index(drop=True)

        # shuffle individual feature
        X_train_c[feature] = X_train_c[feature].sample(frac=1, random_state=random_state).reset_index(
            drop=True)
        # make prediction with shuffled feature and calculate f1 score
        predictions1 = (model.predict_proba(X_train_c)[:, 1] > 0.5).astype(int)
        shuff_f1_score = roc_auc_score(y_train_c, predictions1)

        # save the drop in f1 score
        feature_dict[feature_index] = (train_auc - shuff_f1_score)

    auc_drop = pd.Series(feature_dict).reset_index()
    auc_drop.columns = ['feature_index', 'f1_score_drop']
    auc_drop.sort_values(by=['f1_score_drop'], ascending=False, inplace=True)
    selected_features_index = auc_drop[auc_drop.f1_score_drop > 0]['feature_index']

    return auc_drop, selected_features_index

final_list = []
folder_path = '../bugdata3'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
for i in range(len(csv_files)):
    list = []
    file_path = csv_files[i]
    list.append(csv_files[i])
    df = pd.read_csv('../bugdata3/'+csv_files[i])
    # 选择特定列
    X_train = df.drop(columns=['bug', 'name'])  # 特征数据，去除目标变量所在列
    y_train = df['bug']  # 目标变量，选择目标变量所在列
    y_train = y_train.apply(lambda x: 1 if x > 0 else x)

    auc,selected_features_index = feature_shuffle_rf(X_train, y_train)
    list1 = selected_features_index.tolist()
    list1 = [x + 1 for x in list1]
    list+= list1
    print(list1)
    final_list.append(list)
print(final_list)
with open('final_auc.txt', 'w') as f:
    for item in final_list:
        f.write("%s\n" % item)

# 打印final_list
