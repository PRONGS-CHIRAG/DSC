#Author: Chirag N Vijay
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("rdata.csv",header=None)
x= data.iloc[:,:-1]
y= data.iloc[:,-1]
lasso=Lasso()
lasso.fit(x,y)
coef=lasso.coef_
print(coef)
