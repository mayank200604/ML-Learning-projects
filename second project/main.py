# iris_logistic.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model

model= RandomForestClassifier(n_estimators=40)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training accuracy: {train_score}")
print(f"Test accuracy: {test_score}")

# Predict on test set
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Convert to DataFrame for seaborn
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Plot all feature pairs
sn.pairplot(iris_df, hue='species', diag_kind='hist')
plt.suptitle("Iris Dataset - Pair Plot", y=1.02)
plt.show()


# Predict on custom data
test_samples = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 2.8, 4.8, 1.8],
    [5.5, 2.4, 3.7, 1.0],
    [7.0, 3.2, 4.7, 1.4],
    [6.3, 3.3, 6.0, 2.5],
    [5.0, 3.4, 1.5, 0.2],
    [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.1, 1.3],
    [6.9, 3.1, 5.1, 2.3]
]
predictions = model.predict(test_samples)
print("Predicted class labels:", predictions)