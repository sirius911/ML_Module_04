import yaml
import numpy as np
from utils.ft_progress import ft_progress
import itertools


def init_model_yaml(file = 'models.yaml'):
    """
    init the file models.yaml
    with structure of all the models
    ['name'] 
    ['alpha']
    ['iter']
    ['mse']
    ['evol_mse']
    ['polynomes']
    ['total_iter']
    ['lambda']
    """
    try:
        pow = range(1, 4 + 1)   # puissances max du polynome = 4
        combi_polynomes = np.array(list(itertools.product(list(itertools.product(pow)), repeat=3)))
        list_models = []
        print('***init model_yaml ***')
        for _,hypo in zip(ft_progress(range(len(combi_polynomes))), combi_polynomes):
            for lambda_ in np.arange(0.0, 1.0, 0.2, dtype=np.float64):
                models = {}
                models['name'] = f"w{hypo[0][0]}d{hypo[1][0]}t{hypo[2][0]}"
                models['alpha'] = 0.1
                models['iter'] = 200
                polynome = list([int(po[0]) for po in hypo])
                models['polynomes'] =polynome
                models['thetas'] = [1 for _ in range(sum(polynome) + 1)]
                models['mse'] = None
                models['lambda'] = float(lambda_)
                models['total_iter'] = 0
                models['evol_mse'] = []
                list_models.append(models)
        with open(file, 'w') as outfile:
                yaml.dump_all(list_models, outfile, sort_keys=False, default_flow_style=None)
        return True
    except Exception as e:
        print(e)
        return False

def load_model(file = 'models.yaml'):
    """
        loqd the file and return a the list of Model in this file or None
    """
    return_list =[]
    try:
        with open(file, 'r') as infile:
            list_models = yaml.safe_load_all(infile)
            return_list = list(list_models)
        return return_list
    except Exception as e:
        print(e)
        return None

def save_model(outfile = 'models.yaml', list_models = None):
    """
        save in yaml the list models in file
        return True if ok False otherwise
    """
    try:
        with open('models.yaml', 'w') as outfile:
            yaml.dump_all(list_models, outfile, sort_keys=False, default_flow_style=None)
            return True
    except Exception as e:
        print(e)
        return False