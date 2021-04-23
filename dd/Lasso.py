# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:24:50 2021

@author: yukin
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X,y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
lr=LinearRegression().fit(X_train,y_train)
#Lasso回帰　　データセットに適用
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
#alphaを減らした場合
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of feature used: {}".format(np.sum(lasso001.coef_ != 0)))
#alphaを小さくしすぎる
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of feature used: {}".format(np.sum(lasso00001.coef_ != 0)))

ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train,y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test,y_test)))


#Ridge alphaをプロットする
plt.plot(lasso.coef_,'s',label="Ridge alpha=1")
plt.plot(lasso001.coef_,'^',label="Ridge alpha=0.01")
plt.plot(lasso00001.coef_,'v',label="Ridge alpha=0.0001")
#LinearRegressのプロットとタグ漬け
plt.plot(ridge01.coef_,'o',label="Ridge alpha0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.xlabel("Coefficient index")
plt.ylabel("coefficient magnitude")
#中心線
plt.hlines(0,0,len(lr.coef_))
#y軸の範囲
plt.ylim(-25,25)
#グラフの調整
plt.legend()









