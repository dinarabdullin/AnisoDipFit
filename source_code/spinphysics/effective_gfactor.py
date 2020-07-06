'''
Calculate an effective g-factor for the given orientations of the magnetic field
'''

import numpy as np


def effective_gfactor(g, field):
	N = field.shape[0]
	geff = np.zeros(N)
	for i in range(N):
		geff[i] = np.sqrt( (g[0] * field[i][0])**2 + 
						   (g[1] * field[i][1])**2 + 
						   (g[2] * field[i][2])**2 )
	return geff