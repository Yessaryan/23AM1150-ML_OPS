import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Example dataset
X = np.array([
    [2.5, 1.7],
    [1.3, 3.1],
    [3.6, 2.9],
    [1.1, 0.9],
    [3.9, 3.2],
    [2.0, 2.5]
])

y = np.array([0, 0, 1, 0, 1, 1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
