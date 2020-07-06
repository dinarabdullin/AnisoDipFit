'''
Calculate temperature-based weights for different g-values
'''

import math
import numpy as np
from supplement.constants import const


def flip_probabilities(geff, g, B0, T):
	N = geff.shape[0]
	p = np.zeros(N)
	for i in range(N):
		boltzmann = np.exp(-const['bohr_magneton'] * B0 * geff[i] / (const['bolzmann_constant'] * T))
		p[i] = 2 * boltzmann / ((1.0 + boltzmann) * (1.0 + boltzmann))
	return p