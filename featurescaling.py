#Author: Chirag N Vijay
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("rdata.csv",header=None)
x= data.iloc[:,:-1]
y= data.iloc[:,-1]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
lasso=Lasso()
lasso.fit(x_scaled,y)
coef=lasso.coef_
print(coef)
