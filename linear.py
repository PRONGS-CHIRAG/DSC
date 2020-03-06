
#Author: Chirag N Vijay
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
bmi_data = pd.read_csv("bmi.csv")
bmi_model = LinearRegression()
bmi_model.fit(bmi_data[['BMI']],bmi_data[['Life expectancy']])
pred=bmi_model.predict(np.array([[25],[50]]))
print(pred)