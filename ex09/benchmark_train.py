import sys
import pandas as pd
import numpy as np
import yaml
from utils.ft_yaml import init_model_yaml, load_model, save_model
from utils.common import loading, colors
from utils.normalizer import Normalizer
from tqdm import tqdm
from utils.utils_ml import add_polynomial_features, cross_validation
from utils.logisticregression import LogisticRegression as myLR
from utils.metrics import f1_score_


def format(arr: np.ndarray, label: int):
    """
    get an array and a label value, return a copy of array where
    label value in it is equal to 1
    and value different of label is equal to 0
    """
    copy = arr.copy()
    copy[:, 0][copy[:, 0] != label] = -1
    copy[:, 0][copy[:, 0] == label] = 1
    copy[:, 0][copy[:, 0] == -1] = 0
    return copy

def format_all(arr):
    """
    get an array of dimension (M, number of label) in argument representing probability of each label
    return an array of dimension (M, 1), where the best probability is choosen
    """
    result = []
    for _, row in arr.iterrows():
        result.append(row.idxmax())
    result = np.array(result).reshape(-1, 1)
    return result

def one_vs_all(k_folds, lambda_, model):
    result = pd.DataFrame()
    x_train, y_train, x_test, y_test = k_folds
    for zipcode in range(4):
        y_train_f = format(y_train, zipcode)
        theta = np.array(model['thetas']).reshape(-1,1)
        alpha = model['alpha']
        max_iter = model['iter']
        my_lr = myLR(theta, alpha, max_iter, lambda_=lambda_)
        my_lr.fit_(x_train, y_train_f)
        y_hat = my_lr.predict_(x_test)
        result[zipcode] = y_hat.reshape(len(y_hat))
    return f1_score_(y_test, format_all(result))

def main():
    print("Loading models ...")
    try:
    # Importation of the dataset
        bio, citi = loading()

        #init models.yaml
        list_model = []
        try:
            with open('models.yaml', 'r') as stream:
                list_model = list(yaml.safe_load_all(stream))
        except Exception as e:
            if not init_model_yaml():
                sys.exit()
    except Exception :
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()

    #normalise
    scaler_x = Normalizer(bio)
    X = scaler_x.norme(bio)
    Y = citi
    for model in tqdm(list_model, leave=False):
        for lamda_ in np.arange(0, 1, 0.2):
            X_poly = add_polynomial_features(X, model['polynomes'])
            for k_folds in cross_validation(X_poly, Y, K=10):
                f1_score = one_vs_all(k_folds, lamda_, model)
                # print(f1_score)

if __name__ == "__main__":
    print("Benchmar starting ...")
    main()
    print("Good by !")