# Importing required libraries.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Creating an dataframe using pandas.
df= pd.read_csv('Insurance.csv')

# Plottting a scatter chart.
plt.scatter(df.Age,df.Insurance,marker='+')
plt.xlabel('Age')
plt.ylabel('Insurance')
plt.title('Age vs Insurance Purchase')
plt.show()

# Creating an train and test split.
X_train,X_test,Y_train,Y_test= train_test_split(df[['Age']],df.Insurance,test_size=0.2,random_state=42)

# Creating a logistic regression.
model= LogisticRegression()
model.fit(X_train,Y_train)

# Predicting.
predictions= model.predict(X_test)
print(f"predictions:{predictions}")

# Checking the accuracy of model.
accuracy= model.score(X_test,Y_test)
print(f"Accuracy is:{accuracy}")

# Predicting the probablity.
propability= model.predict_proba(X_test)
print(f"probability is:{propability}")