# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:23:23 2021

@author: DEVANAND R
"""

import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\16.Multiple Linear Regression\Avacado_Price.csv')

df.isnull().sum() #checking NA values


from sklearn.preprocessing import LabelEncoder



#label encoding for type and year column

lenc = LabelEncoder()

df['type'] = lenc.fit_transform(df['type'])
df['year'] = lenc.fit_transform(df['year'])

#onehot encoding for region column

df = pd.get_dummies(df)



sns.pairplot(df) #checking colinearity

X = df.drop('AveragePrice', axis = 1)

Y = df['AveragePrice']


from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(X, Y, test_size = 0.2 ,random_state=42)

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
