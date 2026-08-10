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


def macro_precision(cm: np.ndarray) -> float:
    """Mean per-class precision, over the same "classes with >=1 true
    example in this eval set" filter macro_f1/balanced_accuracy use -- a
    class absent from y_true but present in y_pred would otherwise pull the
    macro average around on a class that isn't actually being evaluated
    here."""
    precision = per_class_precision(cm)
    support = cm.sum(axis=1)
    return float(precision[support > 0].mean()) if (support > 0).any() else 0.0


def macro_f1(cm: np.ndarray) -> float:
    p, r = per_class_precision(cm), per_class_recall(cm)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    support = cm.sum(axis=1)
    return float(f1[support > 0].mean()) if (support > 0).any() else 0.0


def majority_vote_by_group(
    group_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapses per-point predictions to one per group (e.g. per cell) by
    majority vote -- matches the SegCLR paper's classifier and this lab's
    replication (aggregation_study/03_train_evaluate.py's
    cell_majority_vote_accuracy): classify each point independently, THEN
    majority-vote for a cell-level answer, never by averaging features
    before classifying (see baseline/mean_pool_classifier.py (deleted 2026-08-06, deprecated cleanup)'s docstring).

    Returns (group_y_true, group_y_pred), one entry per unique group_id in
    the order np.unique returns them -- pass straight to summarize().
    """
    groups = np.unique(group_ids)
    true_out = np.empty(len(groups), dtype=y_true.dtype)
    pred_out = np.empty(len(groups), dtype=y_pred.dtype)
    for i, g in enumerate(groups):
        mask = group_ids == g
        true_out[i] = y_true[mask][0]  # constant within a group by construction
        pred_out[i] = np.bincount(y_pred[mask]).argmax()
    return true_out, pred_out


def summarize(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, classes: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    recall = per_class_recall(cm)
    precision = per_class_precision(cm)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": balanced_accuracy(cm),
        "macro_precision": macro_precision(cm),
        "macro_f1": macro_f1(cm),
        "per_class_recall": {c: float(recall[i]) for i, c in enumerate(classes)},
        "per_class_precision": {c: float(precision[i]) for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
    }
