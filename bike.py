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
from datetime import datetime

#https://www.kaggle.com/c/bike-sharing-demand/submissions?sortBy=date&group=all&page=1
#loading the data
train = pd.read_csv('E:/Python Github/Bike - Kaggle/train.csv')
test = pd.read_csv('E:/Python Github/Bike - Kaggle/test.csv')
full = train.append( test , ignore_index = True )
#obseving the data
train.head()
train.info()
train.describe()

full.info()
full.describe()
train.columns
test.columns

#datetime
datetimeDF=pd.DataFrame()
datetimeDF['datetime']=full['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

datetimeDF['month']=datetimeDF["datetime"].dt.month
datetimeDF['hour']=datetimeDF["datetime"].dt.hour
datetimeDF['day']=datetimeDF["datetime"].dt.day
full = pd.concat([full,datetimeDF],axis=1)
full=full.drop(['casual','registered','datetime'],axis=1)

#delete useless columns
sourceRow=10886
train=full.loc[0:sourceRow-1,:]
test=full.loc[sourceRow:,:]
train.info()

#Correlation
corrDf = full.corr() 
corrDf['count'].sort_values(ascending =False)

#one-hot (unnecessary?)
full.columns
seasonDf = pd.DataFrame()
seasonDf = pd.get_dummies( full['season'] , prefix='season' )
seasonDf.head()
full = pd.concat([full,seasonDf],axis=1)
full.drop('season',axis=1,inplace=True)
full.head()

weatherDf = pd.DataFrame()
weatherDf = pd.get_dummies( full['weather'] , prefix='weather' )
weatherDf.head()
full = pd.concat([full,weatherDf],axis=1)
full.drop('weather',axis=1,inplace=True)
full.head()

full_X=full.drop('count', axis=1)

#model fitting
source_X = full_X.loc[0:sourceRow-1,:]
source_y = full.loc[0:sourceRow-1,'count']  
pred_X = full_X.loc[sourceRow:,:]
source_X.shape[0]

train_X, test_X, train_y, test_y = train_test_split(source_X,source_y,train_size=.8)

#sns.pairplot(train.drop("datetime", axis=1))
#sns.heatmap(train.drop("datetime", axis=1).corr(), linecolor='white', annot=True,vmin=0, vmax=1)
sns.distplot(train["count"])

#Check the score using the training data
rf = RandomForestRegressor()
rf.fit( train_X , train_y )
rf.score(test_X , test_y )
print(rf.score(test_X , test_y ))

#predict
rf2 = RandomForestRegressor()
rf2.fit( source_X,source_y)
pred_y = rf2.predict(pred_X)
pred_y =pred_y.astype(int)
sns.distplot(pred_y)

test_full = pd.read_csv('E:/Python Github/Bike - Kaggle/test.csv')
submission = pd.DataFrame({"datetime": test_full["datetime"], "count": pred_y})
submission=submission[["datetime","count"]]
submission.shape
print(submission.head())
submission.to_csv("submission.csv", index=False)

