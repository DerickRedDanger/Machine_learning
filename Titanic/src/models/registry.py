# src/models/registry.py

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


MODEL_REGISTRY = {
    "logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "tree": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
}

if XGBClassifier is not None:
    MODEL_REGISTRY["xgb"] = XGBClassifier