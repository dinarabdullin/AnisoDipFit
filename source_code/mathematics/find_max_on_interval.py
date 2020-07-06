'''
Find the maximal value of y(x) on the interval [xmin, xmax]
'''

import numpy as np


def find_max_on_interval(y, x, xMin, xMax):
    ys = []
    for i in range(x.size):
        if (x[i] >= xMin) and (x[i] <= xMax):
            ys.append(y[i])
    ysMax = max(ys)	
    return ysMax