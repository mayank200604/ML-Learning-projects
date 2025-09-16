# Importing all the required libraries.
import pandas as pd
import numpy as np
from matplotlib import pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating a dataframe using pandas.
df= pd.read_csv('HR-Employee-Attrition.csv')

# Getting the genral information og the csv file.
print("The info is:")
print(df.info())
print("The describe is:")
print(df.describe())
print("The value counts of Attributes are:")
print(df['Attrition'].value_counts())

# Using labelencoder.
le= LabelEncoder()
df['Attrition']= le.fit_transform(df['Attrition'])
print(df['Attrition'])
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis='columns')
X = df.drop('Attrition', axis='columns')
y = df['Attrition']

# Training and splitting process.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using scaller.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Setting up a logistic regression.
model=LogisticRegression(class_weight='balanced' ,max_iter=500)
model.fit(X_train,y_train)

Y_pred= model.predict(X_test)
print(Y_pred)

# Using confusion matrix and plotting graph.
cm = confusion_matrix(y_test, Y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Accuracy and precision.
print("Accuracy:", accuracy_score(y_test, Y_pred))
print("Classification Report:\n", classification_report(y_test, Y_pred))


