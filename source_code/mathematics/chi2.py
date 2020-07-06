'''
Calculate chi2 between two signals
'''

import numpy as np


def chi2(ys, ye, x, x_min, x_max, sn=0):
    chi2 = 0.0
    norm = 1.0
    if sn:
        norm = 1 / sn**2
    for i in range(ye.size):
        if (np.absolute(x[i]) >= x_min) and (np.absolute(x[i]) <= x_max):
            chi2 += norm * (ys[i] - ye[i])**2
    return chi2