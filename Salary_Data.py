# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 14:32:46 2021

@author: DEVANAND R
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\15.Simple linear Regression\Salary_Data.csv')



plt.scatter(df.YearsExperience, df.Salary)

np.corrcoef(df.YearsExperience, df.Salary)

# Simple Linear Regression
import statsmodels.formula.api as smf

model = smf.ols('Salary ~ YearsExperience', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df.YearsExperience))

# Regression Line
import seaborn as sns

sns.regplot(x=df.Salary, y=pred1, data=df)

# Error calculation
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(df.Salary, pred1)
rmse = np.sqrt(mse1)
rmse

#Model building on Transformed Data
# Log Transformation



model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df.YearsExperience))

# Regression Line
sns.regplot(x=df.Salary, y=pred2, data=df)

#rmse

mse2 = mean_squared_error(df.Salary, pred2)

rmse2 = np.sqrt(mse2)
rmse2



#EXPONENTIAL TRANSFORMATION


model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df.YearsExperience))
pred3_conv = np.exp(pred3)


# Regression Line
sns.regplot(x=df.Salary, y=pred3_conv, data=df)


mse3 = mean_squared_error(df.Salary, pred3_conv)

rmse3 = np.sqrt(mse3)
rmse3

