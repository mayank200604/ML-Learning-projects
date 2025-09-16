#importing needed libraries.
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

#taking the data from the respected file using panda dataframe.
df = pd.read_csv("prices.csv")

median_bedroom= math.floor(df.Bedroom.median())

df.Bedroom= df.Bedroom.fillna(median_bedroom)

#Creating a linearregression model.
reg= linear_model.LinearRegression()
print(reg.fit(df[['Area','Bedroom','Age']],df.Price))

print(reg.coef_)
print(reg.intercept_)
inputs= [[3000,3,40],
         [1600,2,8],
         [4500,4,22]]
predicted= reg.predict(inputs)
print("The predicted values are :",predicted)

#Making a scatter plot for the regression.
X = df[['Area', 'Bedroom', 'Age']]
y = df['Price']
model = linear_model.LinearRegression()
model.fit(X, y)
plt.scatter(df['Area'], df['Price'], color='blue', label='Actual Data')
plt.xlabel('Area (sq.ft)')
plt.ylabel('Price')
plt.title('Linear Regression: Area vs Price')
plt.legend()
plt.show()