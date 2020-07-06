'''
Get the optimized values of fitting parameters
'''

import copy
import numpy as np
from supplement.constants import const


def get_parameters(genes, indices, fixed, errors=[]):
    parameters = {}
    for name in const['variable_names']:
        index = indices[name]
        if not (index == -1):
            value = genes[index]
            optimized = 'Y'
            if name in errors:
                precision = errors[name]
            else:
                precision = np.nan
        else:
            value = fixed[name]
            optimized = 'N'
            precision = np.nan
        parameter = {}
        parameter['longname'] = const['long_variable_names'][name]
        parameter['value'] = value
        parameter['optimized'] = optimized
        parameter['precision'] = precision
        parameters[name] = parameter
    return parameters