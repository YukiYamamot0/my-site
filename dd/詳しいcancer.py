# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:55:59 2021

@author: yukin
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#ロジスティック回帰
from sklearn.linear_model import LogisticRegression
#線形サポートベクタマシン
from sklearn.svm import LinearSVC

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train,y_train)

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)

for C, marker in zip([0.001,1,100],['o','^','v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1",solver='liblinear').fit(
        X_train,y_train)
    print("Training accurary of l1 logreg with C ={:.3f}: {:.2f}".format(
        C,lr_l1.score(X_train,y_train)))
    print("Test accurary of l1 logreg with C ={:.3f}: {:.2f}".format(
        C,lr_l1.score(X_test,y_test)))
    plt.plot(lr_l1.coef_.T, marker,label="C={:.3f}".format(C))

#範囲決め
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.hlines(0,0,cancer.data.shape[1])
plt.xlabel("Feature")
plt.ylabel("Coefficient magntitude")
 ## 訓練セットに対する精度(C=0.001): 0.91
 ## テストセットに対する精度(C=0.001): 0.92
 ## 訓練セットに対する精度(C=1.000): 0.96
 ## テストセットに対する精度(C=1.000): 0.96
 ## 訓練セットに対する精度(C=100.000): 0.99
 ## テストセットに対する精度(C=100.000): 0.98
plt.ylim(-5,5)
plt.legend(loc=3)