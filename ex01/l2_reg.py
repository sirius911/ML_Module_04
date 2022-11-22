import numpy as np


def valid(function):
    def validation(*args, **kwargs):
        try:
            name=function.__name__
            theta_,  = args
            if not isinstance(theta_, np.ndarray):
                print("Error in iterative_l2(): Not a numpy.array.")
                return None
            m, n = theta_.shape
            if n != 1:
                print(f"Error in {name}(): bad shape of theta -> {theta_.shape}")
                return None
            if theta_.size == 0:
                print("Error in {name}(): empty theta.")
                return None
            return function(*args, **kwargs)
        except Exception as e:
            print(f"Error in {name}(): {e}")
            return None
    return validation
    
@valid
def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    theta_ = np.copy(theta).astype(float)
    sum = 0
    for i in range(1, theta_.size):
        sum += theta_[i][0] ** 2
    return sum

@valid
def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    theta = np.reshape(theta, (len(theta), ))
    return float(theta[1:].T.dot(theta[1:]))

