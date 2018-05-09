# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:05:07 2018

@author: hayas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#loading the data
train = pd.read_csv('E:/Python Github/Bike - Kaggle/train.csv')
test = pd.read_csv('E:/Python Github/Bike - Kaggle/test.csv')
#obseving the data
train.head()
train.info()
train.describe()
#sns.pairplot(train.drop("datetime", axis=1))
#sns.heatmap(train.drop("datetime", axis=1).corr(), linecolor='white', annot=True,vmin=0, vmax=1)
sns.distplot(train["count"])

x_train=train.drop(['casual','registered','count','datetime'],axis=1)
x_test=test.drop('datetime',axis=1)
y_train=train["count"]
y_train_log=np.log(y_train)

rf = RandomForestRegressor()
rf.fit(x_train, y_train_log)
prediction_log = rf.predict(x_test)
prediction=np.exp(prediction_log)
sns.distplot(prediction)



submission = pd.DataFrame({"datetime": test["datetime"], "count": prediction})
submission.to_csv("submission.csv", index=False)

