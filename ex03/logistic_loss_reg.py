import numpy as np


def valid(function):
    def validation(*args, **kwargs):
        try:
            name=function.__name__
            y, y_hat, theta, lambda_,  = args
            if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or not isinstance(theta, np.ndarray):
                print(f"Error in {name}(): y, y_hat or theta not numpy.array.")
                return None
            if not isinstance(lambda_, float):
                print(f"Error in {name}(): lamba_ must be a float.")
                return None
            if y.shape[1] != y_hat.shape[1] or y_hat.shape[1] != theta.shape[1] or theta.shape[1] != 1:
                print(f"Error in {name}(): bad shape")
                return None
            return function(*args, **kwargs)
        except Exception as e:
            print(f"Error in {name}(): {e}")
            return None

    return validation

def regularize(function):
    def regul(*args, **kwargs):
        loss = function(*args, **kwargs)
        y, _, theta, lambda_,  = args
        m = y.shape[0]
        t_ = np.squeeze(theta[1:])
        reg = lambda_ * t_ @ t_ / (2 * m)
        return -loss / m + reg
    
    return regul

@valid
@regularize
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
    eps = 1e-15
    y_ = np.squeeze(y)
    y_hat_ = np.squeeze(y_hat)
    loss = y_ @ np.log(y_hat_ + eps) + (1 - y_) @ np.log(1 - y_hat_ + eps)
    return loss