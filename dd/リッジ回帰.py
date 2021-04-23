# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:47:47 2021

@author: yukin
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


from sklearn.linear_model import Ridge
X,y=mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
lr=LinearRegression().fit(X_train,y_train)

#傾きを小さくしたい。
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test,y_test)))
#訓練セットとテストセットのスコアを比較する。alphaを大きくする。 
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test,y_test)))
#alphaを小さくする。
ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train,y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test,y_test)))

#Ridge alphaをプロットする
plt.plot(ridge.coef_,'s',label="Ridge alpha=1")
plt.plot(ridge10.coef_,'^',label="Ridge alpha=10")
plt.plot(ridge01.coef_,'v',label="Ridge alpha=0.1")
#LinearRegressのプロットとタグ漬け
plt.plot(lr.coef_,'o',label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("coefficient magnitude")
#中心線
plt.hlines(0,0,len(lr.coef_))
#y軸の範囲
plt.ylim(-25,25)
#グラフの調整
plt.legend()
#学習曲線
mglearn.plots.plot_ridge_n_samples()