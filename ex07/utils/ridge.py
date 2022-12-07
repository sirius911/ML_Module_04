import numpy as np
from utils.mylinearregression import MyLinearRegression
from utils.ft_progress import ft_progress

def control(function):

    def essais(*args, **kwargs):
        name = function.__name__
        try:
            return function(*args, **kwargs)
        except Exception as e:
            print(f"Error in {name}(): {e}")
            return None
    
    return essais

class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5, progress_bar=True):
        super(MyRidge, self).__init__(thetas=thetas, alpha=alpha, max_iter=max_iter, progress_bar=progress_bar)
        self.lambda_ = lambda_

    def get_params_(self):
        """
            return theta, alpha, max_iter, lambda_, progress_bar
        """
        return (self.thetas, self.alpha, self.max_iter, self.lambda_, self.progress_bar)
    
    @control
    def set_params_(self, theta=None, alpha=None, max_iter=None, lambda_=None, progress_bar=None ):
        """
        set the parameters to MyRidge object
        args:
            (thetas, alpha, max_iter, lambda_, progress_bar)
        """
        if theta is not None:
            self.thetas = theta
        if alpha is not None:
            self.alpha = alpha
        if max_iter is not None:
            self.max_iter = max_iter
        if lambda_ is not None:
            self.lambda_ = lambda_
        if progress_bar is not None:
            self.progress_bar = progress_bar

    def l2(self):
        """Computes the L2 regularization of a non-empty numpy.array, without any
        for-loop.
        Args:
            theta: has to be a numpy.array, a vector of shape n’ * 1.
        Return:
            The L2 regularization as a float.
            None if theta in an empty numpy.array.
            None if theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        theta = np.reshape(self.thetas, (len(self.thetas), ))
        return float(theta[1:].T.dot(theta[1:]))   

    @control
    def loss_(self, y, y_hat):
        """
        return the loss between 2 vectors (numpy arrays),
        """
        m = y.shape[0]
        t_ = np.squeeze(self.thetas[1:])
        loss = (y - y_hat).T @ (y - y_hat)
        reg = self.lambda_ * t_ @ t_
        return float(0.5 * (loss + reg) / m)

    @control
    def loss_elem_(self, y, y_hat):
        loss = (y - y_hat) ** 2
        reg = self.lambda_ * self.l2()
        return (loss+reg)

    @control
    def gradient_(self, x, y):
        """ calculates the vectorized regularized gradient"""
        m = x.shape[0]
        y_hat = self.predict_(x)
        x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
        diff = y_hat - y
        t_ = np.copy(self.thetas)
        t_[0] = 0
        return ((1 / m) * (x.T.dot(diff) + (self.lambda_ * t_)))

    @control
    def fit_(self, x, y):
        """fits Ridge regression model to a training dataset."""
        list = range(self.max_iter)
        if self.progress_bar:
            list = ft_progress(list)
            list_mse = []
            for _ in list:
                gradien = self.gradien_(x, y)
                self.thetas = self.thetas - (self.alpha * gradien)
                mse = MyLinearRegression.mse_(y, self.predict_(x))
                list_mse.append(mse)
        return list_mse
