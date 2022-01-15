# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 13:39:22 2021

@author: DEVANAND R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\15.Simple linear Regression\delivery_time.csv')

df.rename(columns = {'Delivery Time':'Dtime', 'Sorting Time':'Stime',}, inplace = True)

plt.scatter(df.Dtime, df.Stime)

np.corrcoef(df.Dtime, df.Stime)

# Simple Linear Regression
import statsmodels.formula.api as smf

model = smf.ols('Dtime ~ Stime', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['Stime']))

# Regression Line
import seaborn as sns

sns.regplot(x=df.Dtime, y=pred1, data=df)

# Error calculation
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(df.Dtime, pred1)
rmse = np.sqrt(mse1)
rmse

#Model building on Transformed Data
# Log Transformation

plt.scatter(df.Dtime, np.log(df.Stime))


model2 = smf.ols('Dtime ~ np.log(df.Stime)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['Stime']))

# Regression Line
sns.regplot(x=df.Dtime, y=pred2, data=df)

#rmse

mse2 = mean_squared_error(df.Dtime, pred2)

rmse2 = np.sqrt(mse2)
rmse2

#EXPONENTIAL TRANSFORMATION


model3 = smf.ols('np.log(df.Dtime) ~ df.Stime', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df.Stime))
pred3_conv = np.exp(pred3)


# Regression Line
sns.regplot(x=df.Dtime, y=pred3_conv, data=df)


mse3 = mean_squared_error(df.Dtime, pred3_conv)

rmse3 = np.sqrt(mse3)
rmse3

#polynomial transformation



model4 = smf.ols('np.log(Dtime) ~ Stime + I(Stime*Stime)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_at = np.exp(pred4)
pred4_at

# Regression line

sns.regplot(x=df.Dtime, y=pred4_at, data=df)

# Error calculation

mse4 = mean_squared_error(df.Dtime, pred4_at)

rmse3 = np.sqrt(mse3)
rmse3
