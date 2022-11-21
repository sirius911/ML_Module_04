import numpy as np


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
    if not isinstance(theta, np.ndarray):
        print("Error in iterative_l2(): Not a numpy.array.")
        return None
    try:
        m, n = theta.shape
        if n != 1:
            print(f"Error in iterative_l2(): bad shape of theta -> {theta.shape}")
            return None
        if theta.size == 0:
            print("Error in iterative_l2(): empty theta.")
            return None
        theta_ = np.copy(theta).astype(float)
        sum = 0
        for i in range(1, theta_.size):
            sum += theta_[i][0] ** 2
        return sum
    except Exception as e:
        print(f"Error in iterative_l2(): {e}")
        return None


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
    try:
        theta = np.reshape(theta, (len(theta), ))
        return float(theta[1:].T.dot(theta[1:]))
    except Exception as inst:
        print(inst)
        return None
