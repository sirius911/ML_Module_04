import sys
import pandas as pd
import numpy as np
import os
from utils.ft_yaml import load_model
from utils.common import colors, loading

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
    print(best_model)

if __name__ == "__main__":
    print("Solar system census starting ...")
    main()
    print("Good by !")