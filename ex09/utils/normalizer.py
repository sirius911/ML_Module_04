import numpy as np


class Normalizer():
    def __init__(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        pass
        
    def norme(self, X):
        X_tr = np.copy(X)
        X_tr -= self.mean_
        X_tr /= self.std_
        return X_tr