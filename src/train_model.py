"""
train_model.py
==============
Standalone training script for the Decision Tree Classifier on the
Smart Grid Stability Augmented dataset.

Run from the project root:
    python src/train_model.py

Outputs evaluation metrics to the console.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
import matplotlib.pyplot as plt
from sklearn import tree


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "smart_grid_stability_augmented.csv")

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Prepare Features and Labels
# ---------------------------------------------------------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

le = LabelEncoder()
y = le.fit_transform(y)

# ---------------------------------------------------------------------------
# 3. Train / Test Split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Hyperparameter Tuning with GridSearchCV
# ---------------------------------------------------------------------------
param_grid = {
    "max_depth": [3, 5, 7, 9],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2"],
    "ccp_alpha": [0.0, 0.001, 0.01],
}

dt = DecisionTreeClassifier(random_state=1024)
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"[INFO] Best Parameters : {grid_search.best_params_}")
print(f"[INFO] Best CV Score   : {grid_search.best_score_:.6f}")

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.6f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted'):.6f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted'):.6f}")
print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted'):.6f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------------
# 6. Tree Visualization
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

dot_path = os.path.join(OUTPUT_DIR, "best_tree.dot")
export_graphviz(
    best_model,
    out_file=dot_path,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
)
print(f"\n[INFO] Decision tree exported to: {dot_path}")
