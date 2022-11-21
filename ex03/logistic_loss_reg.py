import numpy as np


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for l
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    try:
        m = y.shape[0]
        eps = 1e-15
        t_ = np.squeeze(theta[1:])
        y_ = np.squeeze(y)
        y_hat_ = np.squeeze(y_hat)
        loss = y_ @ np.log(y_hat_ + eps) + (1 - y_) @ np.log(1 - y_hat_ + eps)
        reg = lambda_ * t_ @ t_ / (2 * m)
        return -loss / m + reg
    except Exception as inst:
        print(inst)
        return None

