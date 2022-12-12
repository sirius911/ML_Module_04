import sys
import numpy as np
import pandas as pd


_NUMERIC_KINDS = set('buifc')

class colors:
    green = '\033[92m' # vert
    blue = '\033[94m' # blue
    yellow = '\033[93m' # jaune
    red = '\033[91m' # rouge
    reset = '\033[0m' #gris, couleur normales

def loading():
    try:
        bio_data = pd.read_csv("solar_system_census.csv", dtype=np.float64)[['weight', 'height', 'bone_density']].values
        citizens = pd.read_csv("solar_system_census_planets.csv", dtype=np.float64)[['Origin']].values.reshape(-1, 1)
    except Exception as e:
        print(e)
        sys.exit(1)

    return bio_data, citizens

def is_numeric(array: np.ndarray):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.
    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS