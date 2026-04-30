import numpy as np


IGNORE_INDEX = -100


def compute_accuracy_metrics(eval_pred):
    """
    Compute token-level accuracy while ignoring labels equal to -100.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != IGNORE_INDEX

    correct = predictions[mask] == labels[mask]
    accuracy = correct.mean() if correct.size > 0 else 0.0

    return {"accuracy": accuracy}