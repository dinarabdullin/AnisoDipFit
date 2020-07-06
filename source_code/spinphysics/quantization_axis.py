'''
Calculate the quantization axis of a spin center for the given orientations of the magnetic field
'''

import numpy as np


def quantization_axis(g, geff, field):
	N = field.shape[0]
	qa = np.zeros((N,3))
	for i in range(N):
		qa[i][0] = (g[0] / geff[i])**2 * field[i][0]
		qa[i][1] = (g[1] / geff[i])**2 * field[i][1]
		qa[i][2] = (g[2] / geff[i])**2 * field[i][2]
	return qa