import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        print("Error in add_polymonial_features_extended(): not numpy.array or int in arguments.")
        return None
    try:
        typ = x.dtype
        m, n = x.shape
        
        if power == 0:
            return np.ones((x.shape)).astype(typ)

        ret = np.zeros((m, n * power))

        #first column
        for i in range(n):
            ret[:,i] = x[:,i]
        # next columns
        idx = 2
        for p in range(2, power + 1):
            for i in range(n):
                ret[:,idx] = x[:,i] ** p
                idx += 1
        return ret.astype(typ)
    except Exception as e:
        print(f"Error in add_polynomial_features: {e}")
        return None