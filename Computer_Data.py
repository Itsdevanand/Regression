# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 20:54:33 2021

@author: DEVANAND R
"""

import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\16.Multiple Linear Regression\Computer_Data.csv')

df.isnull().sum() #checking NA values

sns.pairplot(df)

Y = df.price

X = df.drop('price', axis =1 )
X = df.drop(df.iloc[:,0:1], axis =1 )

from sklearn.preprocessing import LabelEncoder

l1 = LabelEncoder()
l2 = LabelEncoder()
l3 = LabelEncoder()
X.cd = l1.fit_transform(X.cd)
X.multi = l2.fit_transform(X.multi)
X.premium = l3.fit_transform(X.premium)

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
