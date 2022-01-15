# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:51:38 2021

@author: DEVANAND R
"""

import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv(r'C:\Users\DEVANAND R\Desktop\Data Science Assignments\16.Multiple Linear Regression\ToyotaCorolla.csv', encoding= 'unicode_escape')

df.isnull().sum() #checking NA values



#taking the columns mentioned in the question

df1['Price'] = df['Price']
df1['Age_08_04'] = df['Age_08_04']
df1['KM'] = df['KM']
df1['HP'] = df['HP']
df1['cc'] = df['cc']
df1['Doors'] = df['Doors']
df1['Gears'] = df['Gears']
df1['Quarterly_Tax'] = df['Quarterly_Tax']
df1['Weight'] = df['Weight']

df = df1

sns.pairplot(df)

Y = df.Price

X = df.drop('Price', axis =1 )


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
model.score(x_test, y_test) #test aaccuracy .


