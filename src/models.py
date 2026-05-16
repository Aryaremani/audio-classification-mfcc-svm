"""
models.py — Stage 3: Model Definitions (MULTICLASS — 3 audio types)
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import SVM_PARAMS, KNN_PARAMS, GB_PARAMS


def build_svm() -> SVC:
    model = SVC(**SVM_PARAMS)
    print(f"  🔵 SVM created (kernel={SVM_PARAMS['kernel']}, C={SVM_PARAMS['C']}, "
          f"multiclass=ovr, 3 classes)")
    return model


def build_knn() -> KNeighborsClassifier:
    model = KNeighborsClassifier(**KNN_PARAMS)
    print(f"  🟡 KNN created (k={KNN_PARAMS['n_neighbors']}, "
          f"metric={KNN_PARAMS['metric']}, 3 classes)")
    return model


def build_gradient_boosting() -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(**GB_PARAMS)
    print(f"  🚀 Gradient Boosting created "
          f"(n_estimators={GB_PARAMS['n_estimators']}, 3 classes)")
    return model
