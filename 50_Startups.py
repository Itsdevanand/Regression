# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 20:05:17 2021

@author: DEVANAND R
"""

import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\16.Multiple Linear Regression\50_Startups.csv')

df.isnull().sum() #checking NA values

sns.pairplot(df)

Y = df.Profit

X = df.drop('Profit', axis =1 )

X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(X, Y, test_size = 0.2 )

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test, y_pred) #r2 score

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)
rmse

model.score(x, y) #train accuracy
model.score(x_test, y_test) #test aaccuracy 

