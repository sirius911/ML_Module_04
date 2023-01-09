import sys
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils.ft_yaml import load_model
from utils.common import colors, loading
from utils.normalizer import Normalizer
from utils.utils_ml import add_polynomial_features, cross_validation, data_spliter
from utils.mylogisticregression import MyLogisticRegression as myLR
from utils.metrics import f1_score_
from matplotlib import pyplot as plt

def display3D(x, y, pred, title):
    error = np.mean(y != pred) * 100
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = {0: "b", 1: "r", 2: "g", 3: "cyan"}
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=pd.DataFrame(y, columns=["Origin"])["Origin"].map(colors), label="true value")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker="x", c=pd.DataFrame(pred, columns=["Origin"])["Origin"].map(colors), label="predicted value")
    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title("{:.2f}%error - {}".format(error, title))
    ax.legend()

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
        model['thetas'][zipcode] = [float(tta) for tta in my_lr.theta]
    return f1_score_(y_test, format_all(result))

def train_with_diff_lambda(X, Y, model):
    lambda_tab = []
    f1_tab = []
    X_poly = add_polynomial_features(X, model['polynomes'])
    print(f" training {colors.yellow}{model['name']}{colors.reset}:")
    for lambda_ in np.arange(0.0, 1.2, 0.2):
        lambda_ = round(lambda_, 1)
        k_folds = data_spliter(X_poly, Y, 0.8)
        f1_score = one_vs_all(k_folds, model, lambda_=lambda_)
        print(f"\twith  λ = {colors.blue}{lambda_:.1f}{colors.reset} f1 score = {colors.green}{f1_score}{colors.reset}")
        lambda_tab.append(lambda_)
        f1_tab.append(f1_score)
    return  lambda_tab, f1_tab

def train(X, Y, model):
    X_poly = add_polynomial_features(X, model['polynomes'])
    k_folds = data_spliter(X_poly, Y, 0.8)
    f1_score = one_vs_all(k_folds, model, lambda_ = model['lambda'])
    model['f1_score'] = f1_score

def show_model(X, Y, model):
    """ show a model with a range of lambda"""
    lambda_t, f1_t = train_with_diff_lambda(X, Y, model)
    x = np.arange(len(lambda_t))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, f1_t, width)
    ax.set_ylabel('F1 Scores')
    ax.set_xlabel('λ')
    ax.set_xticks(x, lambda_t)
    ax.bar_label(rects1, padding=3)
    fig.tight_layout()
    plt.title(f"Model : {model['name']}")
    # find best lambda
    best_f1 = 0
    best_lambda = model['lambda']
    for l,f in zip(lambda_t, f1_t):
        if f > best_f1:
            best_f1 = f
            best_lambda = l
    model['lambda'] = best_lambda

def display(model, X, Y):
    max_iter = model['iter']
    alpha = model['alpha']
    lamda_ = model['lambda']
    X_poly = add_polynomial_features(X, model['polynomes'])
    pred = pd.DataFrame()
    for zipcode in range(4):
        theta = np.array(model['thetas'][zipcode]).reshape(-1, 1)
        my_lr = myLR(theta=theta, alpha=alpha, max_iter=max_iter, lambda_=lamda_)
        y_hat = my_lr.predict_(X_poly)
        pred[zipcode] = y_hat.reshape(len(y_hat))
    pred = format_all(pred)
    display3D(X, Y, pred, f"Model {model['name']} λ = {model['lambda']:.1f}")
        

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
    show_model(X, Y, best_model)
    print(f"predict data with model : {colors.green}{best_model['name']}{colors.reset} λ = {best_model['lambda']:.1f} f1_score = {best_model['f1_score']}")
    display(best_model, X, Y)
    plt.show()

if __name__ == "__main__":
    print("Solar system census starting ...")
    main()
    print("Good by !")