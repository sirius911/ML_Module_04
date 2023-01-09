import os
import sys
import pandas as pd
import numpy as np
import yaml
from utils.ft_yaml import init_model_yaml, save_model
from utils.common import loading, colors
from utils.normalizer import Normalizer
from tqdm import tqdm
from utils.utils_ml import add_polynomial_features, cross_validation
from utils.mylogisticregression import MyLogisticRegression as myLR
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

def one_vs_all(k_folds, model):
    result = pd.DataFrame()
    x_train, y_train, x_test, y_test = k_folds
    for zipcode in range(4):
        y_train_f = format(y_train, zipcode)
        theta = np.array(model['thetas'][zipcode]).reshape(-1,1)
        alpha = model['alpha']
        max_iter = model['iter']
        lambda_ = model['lambda']
        my_lr = myLR(theta, alpha, max_iter, lambda_=lambda_)
        my_lr.fit_(x_train, y_train_f)
        y_hat = my_lr.predict_(x_test)
        result[zipcode] = y_hat.reshape(len(y_hat))
        model['thetas'][zipcode] = [float(tta) for tta in my_lr.theta]
    return f1_score_(y_test, format_all(result))

def train(X, Y, list_model):

    K = 10 # nb parts in cross_validation
    for idx, model in enumerate(tqdm(list_model, leave=False)):
        # print(f"{idx + 1}/{len(list_model)}Model {model['name']} lambda = {model['lambda']:0.2f} -> f1 score = ", end='\r', flush=True)
        X_poly = add_polynomial_features(X, model['polynomes'])
        model['f1_score'] = 0.0
        for k_folds in tqdm(cross_validation(X_poly, Y, K=K), leave=False):
            f1_score = one_vs_all(k_folds, model)
            model['f1_score'] = model['f1_score'] + f1_score
        #mean f1
        model['f1_score'] = model['f1_score'] / K
    return list_model

def main():
    if os.path.exists('models.yaml') == False:
        if not init_model_yaml():
            sys.exit()
    try:
        print("Loading data ...", end='')
        # Importation of the dataset
        bio, citi = loading()
        #init models.yaml
        list_model = []
        with open('models.yaml', 'r') as stream:
            list_model = list(yaml.safe_load_all(stream))
    except Exception :
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    if list_model is not None and len(list_model) > 0:
        print(colors.green,"ok", colors.reset)
    else:
        print(colors.red, "KO", colors.reset)
        return

    print("Normalize ...", end='')
    #normalise
    scaler_x = Normalizer(bio)
    X = scaler_x.norme(bio)
    Y = citi
    print(colors.green, "ok", colors.reset)

    print("******** TRAINING ********")
    list_model = train(X, Y, list_model)
    print("**************************")

    print("Saving models ...", end='')
    if save_model('models.yaml', list_model):
        print(colors.green, "ok", colors.reset)
    else:
        print(colors.red, "KO", colors.reset)

if __name__ == "__main__":
    print("Benchmar starting ...")
    main()
    print("Good by !")