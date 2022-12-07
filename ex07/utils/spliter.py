import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
        while respecting the given proportion of examples to be kept in the training set.
        Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            y: has to be an numpy.array, a vector of dimension m * 1.
            proportion: has to be a float, the proportion of the dataset that will be assigned to the
                training set.
        Return:
            (x_train, x_test, y_train, y_test) as a tuple of numpy.array
            None if x or y is an empty numpy.array.
            None if x and y do not share compatible dimensions.
            None if x, y or proportion is not of expected type.
        Raises:
            This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        print("x and y must be a numpy.array in data_spliter.")
        return None
    if not isinstance(proportion, float):
        print("proportion must be a float in data_spliter.")
        return None
    try:
        m = x.shape[0]
        n = x.shape[1]
        if m != y.shape[0] or y.ndim != 2 or x.ndim != 2 or y.shape[1] != 1:
            print("incompatible array in data_spliter.")
            return None
        data = np.hstack((x, y)) # association des deux matrices
        np.random.default_rng().shuffle(data) # si on ne veut pas le meme resultat on enleve seed
        p = int(np.floor(x.shape[0] * proportion))
        x_train = data[0:p, :-1]
        x_test = data[p:, :-1]
        y_train = data[:p, -1:]
        y_test = data[p:, -1:]
        return (x_train, x_test, y_train, y_test)
    except Exception as e:
        print(e)
        return None


def K_spliter(x, y, K):
    """
    split data into K parts
    args:
        x and y, numpy array to split
        K: int , nb of parts
    return:
        generator of x_train, y_train, x_test, y_test
        None if error
    """
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
        print("Error in cross_validation() : Not numpy array")
        return None
    if (x.shape[0] != y.shape[0]):
        print("Error in K_splitter(): invalid shape")
        return None

    start = int((1 / K) * len(y))
    for n in range(K):
        x_k = x[(start * n):(start * (n + 1))]
        y_k = y[(start * n):(start * (n + 1))]
        x_train, x_test, y_train, y_test = data_spliter(x_k, y_k, .8)
        yield (x_train, x_test, y_train, y_test)
