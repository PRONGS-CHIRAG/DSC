#Author: Chirag N Vijay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
data = np.asarray(pd.read_csv('ddata.csv',header=None))
X=data[:,0:2]
y=data[:,2]
model = DecisionTreeClassifier()
model.fit(X,y)
y_pred=model.predict(X)
print(y_pred)
acc=accuracy_score(y,y_pred)
print( acc)