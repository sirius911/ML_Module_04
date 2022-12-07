from cmath import inf, nan
import sys
from utils.ft_yaml import init_model_yaml, load_model, save_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.spliter import K_spliter
from utils.polynomial_model import add_polynomial_features
from utils.ridge import MyRidge

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

def graph_3D(data):
    """
    show the graphics of data in 3D
    """
    target = data.target.values.reshape(-1, 1) #price
    weight = data.weight.values.reshape(-1, 1)
    prod_distance = data.prod_distance.values.reshape(-1, 1)
    time_delivery = data.time_delivery.values.reshape(-1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Analyzer')
    ax.set_xlabel('weight')
    ax.set_ylabel('distance')
    ax.set_zlabel('time')
    min_target = target.min()
    max_target = target.max()
    taille = target - min_target
    taille = taille / (max_target - min_target) * 100
    p = ax.scatter(weight, prod_distance, time_delivery, s=taille, c=target, alpha = 0.9, cmap='viridis', vmin = min_target, vmax = max_target)
    cbar = plt.colorbar(p)
    cbar.set_label("price of the order (in trantorian unit)", labelpad=+1)
    plt.show()

def main():
    print("Loading models ...")
    try:
    # Importation of the dataset
        data = pd.read_csv("space_avocado.csv", dtype=np.float64)

        #init models.yaml
        try:
            with open('models.yaml'): pass
        except IOError:
            if not init_model_yaml():
                sys.exit()
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    target = data.target.values.reshape(-1, 1) #price
    Xs = data[['weight','prod_distance','time_delivery']].values # features
    
    # 3D
    # graph_3D(data)

    #Normalizer
    scaler_x = Normalizer(Xs)
    scaler_y = Normalizer(target)
    K = 10 #nb of K-folds
    xN = scaler_x.norme(Xs)
    yN = scaler_y.norme(target)
    print(f"Nb x = {len(xN)} y = {len(yN)}")
    update_list = load_model()
    changed = False
    if update_list is not None:
        print(f"{green}Ok{reset}")
        nb_model = len(update_list)
        for idx, model in zip(range(nb_model), update_list):
            print(f"Model {idx + 1}/{nb_model} -> {yellow}{model['name']}{reset}", end=" ")     
            if model['mse'] is None or float(model['mse']) == inf or np.isnan(float(model['mse'])):
                
                hypo = model['polynomes']
                thetas = model['thetas']
                lambda_ = model['lambda']
                max_iter = model['iter']
                alpha = model['alpha']
                mse_list = []
                historic = []
                print(f" with lambda = {lambda_}... training ", end='')
                for i_k, k_folds in enumerate(K_spliter(xN, yN, K)):
                    print(f" {i_k} / {K}")
                    x_train, x_test, y_train, y_test = k_folds
                    x_train_ = add_polynomial_features(x_train, hypo)
                    x_test_ = add_polynomial_features(x_test, hypo)
                    my_ridge = MyRidge(thetas, alpha, max_iter, lambda_, progress_bar=True)
                    historic = my_ridge.fit_(x_train_, y_train)
                    old_list = model['evol_mse']
                    old_list.extend(historic)
                    model['evol_mse'] = [float(m) for m in old_list]
                    mse = MyRidge.mse_(y_test, my_ridge.predict_(x_test_))
                    print(f"\tMSE = {green}{mse}{reset}")
                    mse_list.append(mse)
                    model['total_iter'] = int(model['total_iter']) + max_iter
                    changed = True
                model['mse'] = float(sum(mse_list) / len(mse_list))
                print(f"Total iter = {model['total_iter']} == {len(model['evol_mse'])}")
            else:
                print(f" with lambda = {model['lambda']:0.2f} ", end='')
                print("MSE = ", end="")
                if model['mse'] is not None or model['mse']!= inf or model['mse']!= nan:
                    print(green, end='')
                else:
                    print(red, end='')
                print(f"{model['mse']:0.8f}{reset}")
    if changed:
        print("Saving models ...", end=" ")
        if save_model(list_models=update_list):
            print(f"{green}OK{reset}")
        else:
            print(f"{red}\tKO{reset}") 

    # controle
    print("model control ...", end='')
    best_model = None
    best_mse = inf
    for model in update_list:
        if model['mse'] is None or float(model['mse']) == inf or np.isnan(float(model['mse'])):
            print(f"Model {yellow}{model['name']}{reset} {red}not good{reset}")
        else:
            if float(model['mse']) < best_mse:
                best_mse = float(model['mse'])
                best_model = model
    print(" ended")
    print(f"Best Model -> {yellow}{best_model['name']}{reset} lambda = {best_model['lambda']:0.2f} with MSE = {green}{best_model['mse']}{reset}")  
    
    #graph mse
    fig = plt.figure()
    ax = fig.add_subplot()
    for model in update_list:
        if model['evol_mse'] is not None and model['name'] == best_model['name']:
            ax.plot(np.arange(len(model['evol_mse'])), np.sqrt(model['evol_mse']), label=model['lambda'])
    ax.set_xlabel("number iteration")
    ax.set_ylabel("$\sqrt{mse}$")
    ax.grid()
    plt.legend(title='$\lambda$')
    plt.title(f"Model : {best_model['name']}")
    plt.show()


if __name__ == "__main__":
    print("Benchmark starting ...")
    main()
    print("Good by!")