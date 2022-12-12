import math
import warnings
import numpy as np

from utils.ft_progress import ft_progress

class MyLogisticException(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """

    supported_penalities = ['l2']
    def __init__(self, theta, alpha=0.001, max_iter=1000, penality='l2', lambda_=1.0, progress_bar=True):
        if (not isinstance(alpha, float) and not isinstance(alpha, int)) or alpha <= 0:
            raise MyLogisticException("MyLogisticException: Alpha must be a float > 0")
        self.alpha = float(alpha)
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise MyLogisticException("MyLogisticException: max_iter must be an int > 0")
        self.max_iter = max_iter
        if isinstance(progress_bar, bool):
            self.progress_bar = progress_bar
        else:
            self.progress_bar = False
        if len(theta) == 0:
            raise MyLogisticException("MyLogisticException: Bad theta")
        if isinstance(theta, np.ndarray):
            self.theta = theta.astype('float64')
        else:
            self.theta = np.array(theta,dtype='float64').reshape(-1,1)
        if not isinstance(lambda_, float) and not isinstance(lambda_, int):
            raise MyLogisticException("MyLogisticException: lambda_ must be a float")
        if penality not in self.supported_penalities:
            penality = None
        self.penality = penality
        self.lambda_ = lambda_ if penality in self.supported_penalities else 0.0

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
        if not isinstance(x, np.ndarray):
            print("Error logistic_predict_(): not numpy arrays.")
            return None
        if x.size == 0:
            print("Error logistic_predict_(): empty array.")
            return None
        try:
            m = x.shape[0]
            n = x.shape[1]
            if self.theta.shape[0] != n + 1 or self.theta.shape[1] != 1:
                print("Error logistic_predict_(): Incompatible shape.")
                return None
            x_ = np.hstack((np.ones((m, 1)), x))
            return MyLogisticRegression.sigmoid_(x_ @ self.theta)

        except Exception as e:
            print(e)
            return None

    def loss_(self, y, y_hat):
        """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        eps=1e-15
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            print("Error in loss_() : not numpy array.")
            return None
        try:
            # m = y.shape[0]
            # if y.shape != y_hat.shape or y.shape[1] != 1:
            #     print("Error in loss_() : incompatible shape.")
            #     return None

            # ones = np.ones(y.shape)
            # inter = (y.T @ np.log(y_hat +eps)) + ((ones - y).T @ np.log(ones - y_hat + eps))
            # return float(-(1 / m) * (inter))
            m = y.shape[0]
            t_ = np.squeeze(self.theta[1:])
            loss = (y - y_hat).T @ (y - y_hat)
            reg = self.lambda_ * t_ @ t_
            return float(0.5 * (loss + reg) / m)
        except Exception as e:
            print(e)
            return None

    def loss_elem_(self, x, y):
        """
        Computes the logistic loss vector.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
        Return:
            The logistic loss vector numpy.ndarray.
            None otherwise.
        Raises:
            This function should not raise any Exception.
        """

        eps = 1e-15
        y_hat = self.predict_(x)
        try:
            return (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        except Exception as e:
            print(f"{e}")
            return None

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y and update theta
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        Returns:
            None
        """
        if not isinstance(x,np.ndarray) or not isinstance(y, np.ndarray):
            print("Error: x or y are not good Numpy.ndarray.")
            return 
        if len(x) == 0 or len(y) == 0:
            print("Error: x or y are empty.")
            return 
        try:
            with warnings.catch_warnings():
                list = range(self.max_iter)
                if self.progress_bar:
                    list = ft_progress(list)
                list_mse = []
                for _ in list:
                    gradien = self.gradien_(x, y)
                    self.theta = self.theta - (self.alpha * gradien)
                    mse = MyLogisticRegression.mse_(y, self.predict_(x))
                    list_mse.append(mse)
                return list_mse
        except Exception as e:
            raise MyLogisticException(e)
    
    def gradien_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have comp
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be an numpy.ndarray, a vector (n +1) * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible shapes.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            print("Error in gradient_(): x is not a numpy array.")
            return None
        if x.size == 0 or y.size == 0:
            print("Error in gradient_(): Empty array")
            return None
        try:
            m, n =x.shape
            if m != y.shape[0] or y.shape[1] != 1 or self.theta.shape[0] != n + 1:
                print("Error in gradient_(): not compatible shape")
                return None
            y_hat = self.predict_(x)
            ones = np.ones(len(x)).reshape(-1, 1)
            x_ = np.concatenate([ones, x], axis=1)
            t_ = np.copy(self.thetas)
            t_[0] = 0
            return (1 / m) * (x_.T.dot(y_hat - y) + self.lambda_ * t_)
        except Exception as e:
            print(f"Error in vec_log_gradient(): {e}")
            return None

    #****************************************************************
    # Class' Methods
    #****************************************************************
    def sigmoid_(x):
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
            The sigmoid value as a numpy.ndarray of shape (m, 1).
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            print("Error in sigmoid(): x is not a numpy array.")
            return None
        if x.size == 0:
            print("Error in sigmoid(): Empty array")
            return None
        try:
            m = x.shape[0]
            n = x.shape[1]
            if n != 1:
                print(f"Error sigmoid(): X.shape = {x.shape} != (m, 1).")
                return None
            ret = np.zeros(x.shape)
            ret = 1 / (1 + np.exp(-x))
            return ret
        except Exception as e:
            print(e)
            return None

    def mse_(y, y_hat):
        """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            loss_elem = (y_hat - y) * (y_hat - y)
            return loss_elem.sum() / len(y)
        except Exception:
            return None

    def rmse_(y, y_hat):
        """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            return math.sqrt(MyLogisticRegression.mse_(y, y_hat))
        except Exception:
            return None
