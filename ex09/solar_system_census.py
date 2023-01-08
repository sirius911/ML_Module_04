import sys
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils.ft_yaml import load_model
from utils.common import colors, loading
from utils.normalizer import Normalizer
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

def one_vs_all(k_folds, model, lambda_):
    result = pd.DataFrame()
    x_train, y_train, x_test, y_test = k_folds
    for zipcode in range(4):
        y_train_f = format(y_train, zipcode)
        polynome = model['polynomes']
        theta = np.array([1 for _ in range(sum(polynome) + 1)]).reshape(-1,1)
        alpha = model['alpha']
        max_iter = model['iter']
        # lambda_ = model['lambda']
        my_lr = myLR(theta, alpha, max_iter, lambda_=lambda_)
        my_lr.fit_(x_train, y_train_f)
        y_hat = my_lr.predict_(x_test)
        result[zipcode] = y_hat.reshape(len(y_hat))
        model['thetas'] = [float(tta) for tta in my_lr.theta]
    return f1_score_(y_test, format_all(result))

def train_with_diff_lambda(X, Y, model):
    
    X_poly = add_polynomial_features(X, model['polynomes'])
    best_lambda = 0
    best_f1 = 0
    for lambda_ in np.arange(0.0, 1.2, 0.2):
        f1_score = 0.0
        for k_folds in tqdm(cross_validation(X_poly, Y, K=10), leave=False):
            f1_score += one_vs_all(k_folds, model, lambda_)
        f1_score /= 10
        model['f1_score'] = f1_score
        print(f"with lambda = {lambda_:.1f} F1 score = {f1_score}")
        if f1_score > best_f1:
            best_f1 = f1_score
            best_lambda = lambda_
    return  best_lambda

def train(X, Y, model):
    X_poly = add_polynomial_features(X, model['polynomes'])
    f1_score = 0.0
    for k_folds in tqdm(cross_validation(X_poly, Y, K=10), leave=False):
        f1_score += one_vs_all(k_folds, model, lambda_ = model['lambda'])
    f1_score /= 10
    model['f1_score'] = f1_score

def main():
    # Importation of the dataset
    print("Loading models ...")
    try:
        if not os.path.exists('models.yaml'):
            print(f"{colors.red}Missing models ...{colors.reset}")
            return
        if not os.path.exists('solar_system_census.csv') or not os.path.exists('solar_system_census_planets.csv'):
            print(f"{colors.red}Missing dataset ...{colors.reset}")
            return
        try:
            print("Loading data ...", end='')
            # Importation of the dataset
            bio, citi = loading()
            
        except IOError as e:
            print(e)
            sys.exit(1)
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    
    update_list = load_model()
    #best score f1
    best_f1 = 0
    best_model = None
    for model in update_list:
        if float(model['f1_score']) > best_f1:
            best_f1 = float(model['f1_score'])
            best_model = model
    # normalise
    scaler_x = Normalizer(bio)
    X = scaler_x.norme(bio)
    Y = citi
    print(colors.green, "ok", colors.reset)

    print("******** TRAINING ********")
    print(f"Training best model : {colors.green}{best_model['name']}{colors.reset} lambda = {best_model['lambda']:.1f} f1 = {best_model['f1_score']}")
    train(X, Y, best_model)
    print("*************")    

if __name__ == "__main__":
    print("Solar system census starting ...")
    main()
    print("Good by !")