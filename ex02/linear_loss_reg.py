import numpy as np


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or not isinstance(theta, np.ndarray):
        print("Error in reg_loss_(): y, y_hat or theta not numpy.array.")
        return None
    if not isinstance(lambda_, float):
        print("Error in reg_loss_(): lamba_ must be a float.")
        return None
    try:
        m = y.shape[0]
        if y.shape[1] != y_hat.shape[1] != theta.shape[1] != 1:
            print("Error in reg_loss_(): bad shape")
            return None
        t_ = np.squeeze(theta[1:])
        loss = (y - y_hat).T @ (y - y_hat)
        reg = lambda_ * t_ @ t_
        return float(0.5 * (loss + reg) / m)
    except Exception as e:
            print(f"Error in reg_loss_(): {e}")
            return None
    