"""Shared helpers: model definitions, preprocessing and metric calculation.
Used by BOTH train_models.py and app.py so the numbers always agree."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN",
               "Naive Bayes", "Support Vector Machine"]
METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def build_preprocessor(X):
    """Impute + scale numeric columns, impute + one-hot encode categoricals."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ])


def build_models(X, seed=42):
    """Return {name: full Pipeline(preprocess + classifier)}."""
    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                                random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                      random_state=seed),
    }
    return {n: Pipeline([("prep", build_preprocessor(X)), ("clf", c)])
            for n, c in clfs.items()}


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Binary -> binary averaging; multi-class -> weighted averaging, OvR AUC."""
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                average="weighted", labels=list(range(n_classes)))
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
