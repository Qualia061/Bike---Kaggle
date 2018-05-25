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
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

#https://www.kaggle.com/c/bike-sharing-demand/submissions?sortBy=date&group=all&page=1
#loading the data
train = pd.read_csv('E:/Python Github/Bike - Kaggle/train.csv')
test = pd.read_csv('E:/Python Github/Bike - Kaggle/test.csv')
train['count']=np.log1p(train['count'])
full = train.append( test , ignore_index = True )

#obseving the data
train.head()
train.info()
train.describe()

full.info()
full.describe()
train.columns
test.columns
print(train.isnull().sum())
print(test.isnull().sum())

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

#Check the score using the training data，RandomForestRegressor
params={'n_estimators':[x for x in range(50,200,20)]}
gbr_best = RandomForestRegressor(n_estimators=70,min_samples_leaf=10,max_features=14)
grid = GridSearchCV(gbr_best, params, cv=5)
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_

gbr_best.fit( train_X , train_y )
gbr_best.score(test_X , test_y )

rf = RandomForestRegressor(n_estimators=50)
rf.fit( train_X , train_y )
rf.score(test_X , test_y )
print(rf.score(test_X , test_y ))

#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(learning_rate=0.1)
gbr.fit( train_X , train_y )
gbr.score(test_X , test_y )
print(gbr.score(test_X , test_y ))

params={'n_estimators':[x for x in range(40,60,10)]}
gbr_best = GradientBoostingRegressor(n_estimators=60,learning_rate=0.2,min_samples_split = 90,min_samples_leaf = 30,max_depth = 8,max_features = 14,subsample = 0.85)
grid = GridSearchCV(gbr_best, params, cv=5, scoring="r2")
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_

gbr_best = GradientBoostingRegressor(n_estimators=500,learning_rate=0.2,min_samples_split = 90,min_samples_leaf = 30,max_depth = 8,max_features = 14,subsample = 0.85)
gbr_best.fit( train_X , train_y )
gbr_best.score(test_X , test_y )
print(gbr_best.score(test_X , test_y))


#KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit( train_X , train_y )
knn.score(test_X , test_y )
print(knn.score(test_X , test_y ))

# XGBoost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier,XGBRegressor

param_test = { 'learning_rate':[i/100.0 for i in range(1,20,2)]}
xgb_best = XGBRegressor(
 learning_rate =0.05,
 n_estimators=160,
 max_depth=6,
 min_child_weight=3,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.7,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
grid = GridSearchCV(estimator = xgb_best, param_grid = param_test, cv=5)
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_

xgb_best.fit( train_X , train_y )
xgb_best.score(test_X , test_y )
print(xgb_best.score(test_X , test_y))

xgb_param =xgb_best.get_xgb_params()
xgb.cv(xgb_param, xgtrain, num_boost_round=5000, nfold=15, metrics=['auc'],
     early_stopping_rounds=50, stratified=True, seed=1301)

full_xy=pd.concat([source_X,source_y],axis=1)
target = 'count'
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=None)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

predictors = [x for x in full_xy.columns if x not in ['count']]
xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, full_xy, predictors)

xgb_param = xgb1.get_xgb_params()
xgtrain = xgb.DMatrix(full_xy[predictors].values, label=full_xy[target].values)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,metrics='rmse', early_stopping_rounds=50)

#predict
rf2 = RandomForestRegressor()

rf2 = GradientBoostingRegressor(n_estimators=60,learning_rate=0.2,min_samples_split = 90,min_samples_leaf = 30,max_depth = 8,max_features = 14,subsample = 0.85)

rf2=RandomForestRegressor(n_estimators=170,min_samples_leaf=10,max_features=14)

rf2= XGBRegressor(
 learning_rate =0.05,
 n_estimators=160,
 max_depth=6,
 min_child_weight=3,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.7,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

rf2.fit( source_X,source_y)
pred_y = rf2.predict(pred_X)

pred_y=np.exp(pred_y)-1
pred_y =pred_y.astype(int)

sns.distplot(pred_y)

test_full = pd.read_csv('E:/Python Github/Bike - Kaggle/test.csv')
submission = pd.DataFrame({"datetime": test_full["datetime"], "count": pred_y})
submission=submission[["datetime","count"]]
submission.shape
print(submission.head())
submission.to_csv("submission.csv", index=False)

