# src/models/registry.py

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


MODEL_REGISTRY = {
    "logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
}

if XGBClassifier is not None:
    MODEL_REGISTRY["xgb"] = XGBClassifier

'''
| Model                  | Family              | Key Idea                     |
| ---------------------- | ------------------- | ---------------------------- |
| LogisticRegression     | Linear              | Linear decision boundary     |
| KNeighborsClassifier   | Instance-based      | Predict from nearby examples |
| SVC                    | Kernel-based        | Maximum-margin classifier    |
| DecisionTreeClassifier | Tree                | Recursive if/then rules      |
| RandomForestClassifier | Bagging Ensemble    | Many trees + voting          |
| ExtraTreesClassifier   | Randomized Ensemble | Extremely randomized trees   |
| XGBClassifier          | Boosting Ensemble   | Sequential error correction  |
'''