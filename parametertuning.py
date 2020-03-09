#Author: Chirag N Vijay
from sklearn.model_selection import GridSearchCV
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.metrics import make_scorer,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
datas=load_breast_cancer()
parameters={'kernel':['poly','rbf'],'C':[0.1,1,10]}
scorer=make_scorer(f1_score)
clf=SVC()
X=pd.DataFrame(datas.data,columns=datas.feature_names)
y=pd.Categorical.from_codes(datas.target,datas.target_names)
ensoder = LabelEncoder()
y=pd.Series(ensoder.fit_transform(y))
grid_obj=GridSearchCV(clf,parameters,scoring=scorer)
grid_fit =grid_obj.fit(X,y)
best_clf=grid_fit.best_estimator_
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
best_clf.predict(X_test)