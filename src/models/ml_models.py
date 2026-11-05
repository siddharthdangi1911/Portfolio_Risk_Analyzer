"""
src/models/ml_models.py
=============================================================================
Core Machine Learning Models: Linear Regression, Decision Tree, SVM, Logistic Regression, LightGBM
Course: ES 335 - Machine Learning
=============================================================================
Provides clean, scikit-learn standard implementations for 1-day risk forecasting:
  1. Linear Regression (L2 Penalized Ridge)
  2. Decision Tree (CART Regressor & Classifier)
  3. Support Vector Machine (SVR & SVC with RBF kernel)
  4. Logistic Regression (Probabilistic Tail Risk Classifier)
  5. LightGBM (Gradient Boosted Trees Benchmark)
"""

import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
import lightgbm as lgb

from src.config import CONFIG

logger = logging.getLogger(__name__)


class LinearRegressionRiskModel:
    """
    Model 1: Linear Regression (Ridge L2 Regularized)
    
    Objective:
      min_w ||y - X w - b||_2^2 + alpha * ||w||_2^2
      
    Interpretable baseline providing linear feature weights (beta coefficients).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.regressor = Ridge(alpha=alpha, random_state=CONFIG.SEED)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.regressor.fit(X_train, y_train)
        logger.info(f"Fitted Linear Regression (Ridge alpha={self.alpha}). Intercept: {self.regressor.intercept_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.regressor.predict(X)

    def get_coefficients(self, feature_names: List[str]) -> Dict[str, float]:
        coefs = self.regressor.coef_
        return {feat: float(round(c, 4)) for feat, c in zip(feature_names, coefs)}


class DecisionTreeRiskModel:
    """
    Model 2: Decision Tree (CART)
    
    Splitting Criterion:
      Variance reduction (MSE) for regression; Gini impurity for classification.
    Non-parametric step-function partitioning of feature space.
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.regressor = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=CONFIG.SEED
        )
        self.classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=CONFIG.SEED
        )

    def fit(self, X_train: np.ndarray, y_reg_train: np.ndarray, y_cls_train: np.ndarray):
        self.regressor.fit(X_train, y_reg_train)
        self.classifier.fit(X_train, y_cls_train)
        logger.info(f"Fitted Decision Tree Regressor & Classifier (max_depth={self.max_depth}).")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_ret = self.regressor.predict(X)
        pred_cls = self.classifier.predict(X)
        pred_prob = self.classifier.predict_proba(X)[:, 1]
        return pred_ret, pred_cls, pred_prob

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        imp = self.regressor.feature_importances_
        norm_imp = imp / np.maximum(1e-8, np.sum(imp))
        return {feat: float(round(score, 4)) for feat, score in zip(feature_names, norm_imp)}


class SVMRiskModel:
    """
    Model 3: Support Vector Machine (SVR & SVC with RBF Kernel)
    
    Maps inputs to infinite-dimensional Hilbert space via Gaussian RBF kernel:
      K(x, x') = exp(-gamma * ||x - x'||^2)
    Maximizes margin epsilon with L2 regularization parameter C.
    """

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, gamma: str = "scale"):
        self.C = C
        self.epsilon = epsilon
        self.regressor = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
        self.classifier = SVC(kernel="rbf", C=C, class_weight="balanced", probability=True, gamma=gamma, random_state=CONFIG.SEED)

    def fit(self, X_train: np.ndarray, y_reg_train: np.ndarray, y_cls_train: np.ndarray):
        self.regressor.fit(X_train, y_reg_train)
        self.classifier.fit(X_train, y_cls_train)
        logger.info(f"Fitted Support Vector Machine (SVR/SVC RBF, C={self.C}, eps={self.epsilon}).")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_ret = self.regressor.predict(X)
        pred_cls = self.classifier.predict(X)
        pred_prob = self.classifier.predict_proba(X)[:, 1]
        return pred_ret, pred_cls, pred_prob


class LogisticRiskClassifier:
    """
    Model 4: Logistic Regression (Probabilistic Risk Classifier)
    
    Formulation:
      P(High Risk = 1 | X) = 1 / (1 + exp(-(w^T x + b)))
      
    Outputs calibrated probabilities of downside crash days with balanced class weighting.
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self.classifier = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=CONFIG.SEED
        )

    def fit(self, X_train: np.ndarray, y_cls_train: np.ndarray):
        self.classifier.fit(X_train, y_cls_train)
        logger.info(f"Fitted Logistic Regression Classifier. Intercept: {self.classifier.intercept_[0]:.4f}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.classifier.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

    def get_coefficients(self, feature_names: List[str]) -> Dict[str, float]:
        coefs = self.classifier.coef_[0]
        return {feat: float(round(c, 4)) for feat, c in zip(feature_names, coefs)}


class LightGBMRiskModel:
    """
    Model 5: Gradient Boosted Trees (LightGBM)
    
    Non-linear tree ensemble optimizing gradient descent in function space.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.03, max_depth: int = 4):
        self.regressor = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=CONFIG.SEED,
            verbose=-1
        )
        self.classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=CONFIG.SEED,
            verbose=-1
        )

    def fit(self, X_train: np.ndarray, y_reg_train: np.ndarray, y_cls_train: np.ndarray):
        self.regressor.fit(X_train, y_reg_train)
        self.classifier.fit(X_train, y_cls_train)
        logger.info("Fitted LightGBM Gradient Boosted Ensembles.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_ret = self.regressor.predict(X)
        pred_cls = self.classifier.predict(X)
        pred_prob = self.classifier.predict_proba(X)[:, 1]
        return pred_ret, pred_cls, pred_prob

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        imp = self.regressor.feature_importances_
        norm_imp = imp / np.maximum(1e-8, np.sum(imp))
        return {feat: float(round(score, 4)) for feat, score in zip(feature_names, norm_imp)}
