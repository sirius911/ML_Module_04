import numpy as np


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_decorator(function):
    def funct(*args, **kwargs):
        ret = function(*args, *kwargs)
        return(sigmoid_(ret))
    return funct

def valid(function):
    def validation(*args, **kwargs):
        try:
            name=function.__name__
            if len(args) == 2:
                x, theta,  = args
                if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
                    print(f"Error in {name}(): x or theta not numpy.array.")
                    return None
                m, n = x.shape
                m_y, n_y = x.shape
            else:
                y, x, theta, lambda_ = args
                if not isinstance(y, np.ndarray):
                    print(f"Error in {name}(): y or theta not numpy.array.")
                    return None
                m, n = x.shape
                m_y, n_y = y.shape
            if m == 0:
                print(f"Error in {name}(): shape[0] of x == 0.")
                return None
            if theta.ndim != 2 and theta.shape[1] != 1:
                print(f"Error in {name}(): shape of theta expected (m, !) != {theta.shape}.")
                return None
            return function(*args, **kwargs)
        except Exception as e:
            print(f"Error in {name}(): {e}")
            return None
    return validation

@sigmoid_decorator  # to compute sigmoid function after the predict function
@valid
def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1,1)
    x_ = np.hstack((np.ones((m, 1)), x))
    y = x_ @ theta
    return y


@valid
def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    gradient = np.zeros((n + 1, 1))
    h = predict_(x, theta)
    
    # [0]
    sum = 0
    for i in range(m):
        sum += (h[i] - y[i])
    gradient[0] = sum / m
    for j in range(1, n + 1):
        sum = 0
        for i in range(m):
            sum += (h[i] - y[i]) * x[i][j - 1]
        gradient[j] = (1 / m) * (sum + lambda_ * theta[j])

    return gradient

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    y_hat = predict_(x, theta)
    x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    diff = y_hat - y
    t_ = np.copy(theta)
    t_[0] = 0
    return ((1 / m) * (x.T.dot(diff) + (lambda_ * t_)))