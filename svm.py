from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
data = np.asarray(pd.read_csv('svm.csv',header=None))
X=data[:,0:2]
y=data[:,2]
model = SVC(kernel='rbf',gamma=30)
model.fit(X,y)
y_pred=model.predict(X)
print(y_pred)
acc=accuracy_score(y,y_pred)
print("Accuracy is")
print(acc)