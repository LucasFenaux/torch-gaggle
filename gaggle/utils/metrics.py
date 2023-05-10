

def accuracy(y_pred, y) -> float:
    """Accuracy metric for classification.

    Args:
        y_pred:
        y:

    Returns:

    """
    return (y_pred.argmax(1) == y).float().mean()
