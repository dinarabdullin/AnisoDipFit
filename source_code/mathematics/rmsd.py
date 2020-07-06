'''
Calculate the root-mean-square-deviation (RMSD) between two signals
'''

import numpy as np


def rmsd(ys, ye, x, x_min, x_max):
    rmsd = 0.0
    count = 0
    for i in range(ye.size):
        if (np.absolute(x[i]) >= x_min) and (np.absolute(x[i]) <= x_max):
            rmsd += (ys[i] - ye[i])**2
            count += 1
    rmsd = np.sqrt(rmsd/float(count))	
    return rmsd