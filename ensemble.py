#Author : Chirag N Vijay
import pandas as pd 
data = pd.read_csv("diabetes.csv")
print(data.shape)
X=data.drop(columns=['Outcome'])
y=data['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y)
#building first model
#knn
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
params_knn={'n_neighbors':np.arange(1,25)}
knn_gs=GridSearchCV(knn,params_knn,cv=5)
knn_gs.fit(X_train,y_train)
knn_best=knn_gs.best_estimator_
print(knn_gs.best_params_)
#building second model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
params_rf={'n_estimators':[50,100,200]}
rf_gs=GridSearchCV(rf,params_rf,cv=5)
rf_gs.fit(X_train,y_train)
rf_best=rf_gs.best_estimator_
print(rf_gs.best_params_)
#building third model
from sklearn.linear_model import LogisticRegression
l=LogisticRegression()
l.fit(X_train,y_train)
print('knn: {}'.format(knn_best.score(X_test,y_test)))
print('rf: {}'.format(rf_best.score(X_test,y_test)))
print('logistic regression: {}'.format(l.score(X_test,y_test)))
#CREATING ENSEMBLE
from sklearn.ensemble import  VotingClassifier
estimators=[('knn',knn_best),('rf',rf_best),('logistic regression',l)]
ensemble= VotingClassifier(estimators,voting='hard')
ensemble.fit(X_train,y_train)
print('ensemble :{}'.format(ensemble.score(X_test,y_test)))
