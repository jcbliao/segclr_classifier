"""Classification metrics beyond raw accuracy -- required whenever classes are
imbalanced (Notion spec: "report balanced accuracy, macro F1, or per-class
recall... raw accuracy alone is not a sufficient result"). Pure numpy, no
sklearn dependency, since these are all small confusion-matrix reductions.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    return np.divide(correct, support, out=np.zeros_like(correct, dtype=np.float64), where=support > 0)


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)
    return np.divide(
        correct, predicted, out=np.zeros_like(correct, dtype=np.float64), where=predicted > 0
    )


def balanced_accuracy(cm: np.ndarray) -> float:
    """Mean per-class recall -- accuracy that doesn't reward always predicting
    the majority class."""
    recall = per_class_recall(cm)
    support = cm.sum(axis=1)
    return float(recall[support > 0].mean()) if (support > 0).any() else 0.0


def macro_f1(cm: np.ndarray) -> float:
    p, r = per_class_precision(cm), per_class_recall(cm)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    support = cm.sum(axis=1)
    return float(f1[support > 0].mean()) if (support > 0).any() else 0.0


def summarize(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, classes: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    recall = per_class_recall(cm)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": balanced_accuracy(cm),
        "macro_f1": macro_f1(cm),
        "per_class_recall": {c: float(recall[i]) for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
    }
