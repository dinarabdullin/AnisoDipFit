'''
Convert fitting parameters into genes
'''

import numpy as np
from supplement.constants import const


def parameters2genes(parameters):
    genes = []
    for name in const['variable_names']:
        if (parameters[name]['optimized'] == 'Y'):  
            genes.append(parameters[name]['value'])
    return np.array(genes)