# Importing the required libraries.
import pandas as pd
import numpy as np
from sklearn import linear_model

# Creating a dataframe using pandas.
df= pd.read_csv('car_price.csv')
df.rename(columns={'Car Model': 'Car_Model'}, inplace=True)

# Using dummies.
dummies= pd.get_dummies(df.Car_Model,dtype=int)

merged=pd.concat([df,dummies],axis='columns')

final= merged.drop(['Car_Model','AUDI AS','Age'],axis='columns')
print("Final dataframe:\n", final)

# Creating and fitting an linear regression.
model= linear_model.LinearRegression()
X= final.drop(['Sell Price'],axis='columns')
Y=df['Sell Price']
model.fit(X,Y)

# Creating an graph.
import matplotlib.pyplot as plt

# Predict using the model
predicted_prices = model.predict(X)

# Scatter plot of actual vs predicted
slope, intercept = np.polyfit(Y, predicted_prices, 1)
line = slope * Y + intercept
plt.scatter(Y, predicted_prices, color='blue', label='Predicted Points')
plt.plot(Y, line, color='green', label='Fit Line')  
plt.xlabel("Actual Sell Price")
plt.ylabel("Predicted Sell Price")
plt.title("Actual vs Predicted Sell Prices")
plt.grid(True)
plt.show()


# Prediction.
inputs= [[45000, 4, 2],
 [60000,0,1]]
predictions= model.predict(inputs)
print("The prediction value is :",predictions)
      
# Accuracy.
accuracy= model.score(X,Y)
print("The accuracy is :",accuracy)