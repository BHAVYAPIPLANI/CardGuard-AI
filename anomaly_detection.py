"""
Anomaly Detection on Credit Card Transactions
Author: Bhavya Piplani

Objective:
Identify fraudulent credit card transactions using Isolation Forest and Local Outlier Factor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
data = pd.read_csv("creditcard.csv")  # Ensure your CSV is in the repo
print(data.head())
print("\nDataset shape:", data.shape)

# Analyze Class Distribution
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]
print(f"\nFraudulent Transactions: {len(fraud)}")
print(f"Normal Transactions: {len(normal)}")

# Prepare Data
X = data.drop('Class', axis=1)
y = data['Class']

# Models for Anomaly Detection
classifiers = {
    "Isolation Forest": IsolationForest(contamination=0.001),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.001)
}

for name, clf in classifiers.items():
    if name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
    else:
        clf.fit(X)
        y_pred = clf.predict(X)
    
    # Convert to 0 and 1
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    print(f"\n{name} Results:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))

# Visualizing correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()
