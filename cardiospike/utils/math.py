import numpy as np
from sklearn.metrics import precision_recall_curve


def threshold_search(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    F = 2 / (1 / (precision + 1e-18) + 1 / (recall + 1e-18))
    F[F > 1.0] = 0
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    return best_th, best_score
