
#Author: Chirag N Vijay
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("data.csv")
x=data['Var_X'].values.reshape(-1,1)
y=data['Var_Y'].values
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
poly_model=LinearRegression(fit_intercept=False).fit(x_poly,y)

#print(res)