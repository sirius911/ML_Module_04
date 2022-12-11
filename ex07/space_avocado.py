from cmath import inf
import os
import sys
import numpy as np
import pandas as pd
from utils.ft_yaml import load_model
from utils.polynomial_model import add_polynomial_features
import matplotlib.pyplot as plt
from utils.ridge import MyRidge

from utils.spliter import data_spliter

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale

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

def main():
    # Importation of the dataset
    print("Loading models ...")
    try:
        if not os.path.exists('models.yaml') or not os.path.exists('space_avocado.csv'):
            print("Missing models or dataset...")
            return
        try:
            with open('models.yaml'): pass
            data = pd.read_csv("space_avocado.csv", dtype=np.float64)
        
        except IOError as e:
            print(e)
            sys.exit(1)
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    
    update_list = load_model()

    #best model
    best_model = None
    best_mse = inf
    best_lambda = None
    for model in update_list:
        if float(model['mse']) is None or float(model['mse']) == inf or np.isnan(float(model['mse'])):
            print(f"Model {yellow}{model['name']}{reset} {red}not good{reset}")
        else:
            if float(model['mse']) < best_mse:
                best_mse = float(model['mse'])
                best_model = model
                best_lambda = model['lambda']
    # Training the best model
    # Importation of the dataset
    target = data.target.values.reshape(-1, 1) #price
    Xs = data[['weight','prod_distance','time_delivery']].values # features

    # split dataset
    x_train, x_test, y_train, y_test = data_spliter(Xs, target.reshape(-1,1), 0.8)

    x_test_to_plot = x_test
    #normalisation
    scaler_x = Normalizer(x_train)
    scaler_y = Normalizer(y_train)

    x = scaler_x.norme(x_train)
    y = y_train #y = scaler_y.norme(y_train)

    x_test = scaler_x.norme(x_test)
    #y_test = scaler_y.norme(y_test)
    
    name = best_model['name']
    polynome = best_model['polynomes']
    thetas = np.array([1 for _ in range(sum(polynome) + 1)],dtype='float64').reshape(-1,1)
    alpha = best_model['alpha']
    # mse_list = best_model['evol_mse']
    mse = best_model['mse']
    iter = best_model['iter']
    lambda_ = best_model['lambda']
    mylr = MyRidge(thetas, alpha, iter, lambda_, progress_bar=True)
    x_ = add_polynomial_features(x, polynome)
    x_test_ = add_polynomial_features(x_test, polynome)

    print(f"The winner is ... : Model {yellow}{name} ({lambda_}){reset} ... training")

    fig, axes = plt.subplots(3, 5, figsize=(15,8))
    for idx, l in enumerate(np.arange(0.0, 1.0, 0.2, dtype=np.float64)):
        mylr.set_params_(theta=thetas, lambda_= l, max_iter=800)
        mylr.fit_(x_, y)

        y_hat = mylr.predict_(x_test_)
        # fig, (ax1, ax2, ax3) = plt.subplots(3)
        titre = f"\lambda = {l} "
        if l == best_lambda:
            titre += " *** BEST ***"
        
        axes[0, idx].scatter(x_test_to_plot[:, 0], y_test, c="b", marker='o', label="price")
        axes[0, idx].scatter(x_test_to_plot[:, 0], y_hat, c='r', marker='x', label="predicted price")
        axes[0, idx].set_title(f"${titre}$")
        axes[1, idx].scatter(x_test_to_plot[:, 1], y_test, c="b", marker='o', label="price")
        axes[1, idx].scatter(x_test_to_plot[:, 1], y_hat, c='r', marker='x', label="predicted price")

        axes[2, idx].scatter(x_test_to_plot[:, 2], y_test, c="b", marker='o', label="price")
        axes[2, idx].scatter(x_test_to_plot[:, 2], y_hat, c='r', marker='x', label="predicted price")
        plt.setp(axes[0, idx], xlabel='weight')
        plt.setp(axes[1, idx], xlabel='distance')
        plt.setp(axes[2, idx], xlabel='time')

    # fig = plt.figure() 11 36%
    # ax = fig.add_subplot()
    # ax.plot(np.arange(iter), (mse_list))
    # plt.title(f"Model : {name} - MSE={mse:0.2e}")
    # ax.set_xlabel("number iteration")
    # ax.set_ylabel("mse")
    # ax.grid()
    plt.setp(axes[0, idx], xlabel='weight')
    plt.setp(axes[1, idx], xlabel='distance')
    plt.setp(axes[2, idx], xlabel='time')
    plt.setp(axes[:, 0], ylabel='price')
    plt.legend()
    plt.subplots_adjust(left=0.04, bottom=0.03, right=0.99, top=0.97, wspace=0.17, hspace=0.13)
    plt.show()

if __name__ == "__main__":
    print("space_avocado starting ...")
    main()
    print("Good by !")