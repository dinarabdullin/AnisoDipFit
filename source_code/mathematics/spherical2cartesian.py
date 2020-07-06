'''
Convert spherical coordinates into cartestian coordinates
'''

import numpy as np


def spherical2cartesian(r, theta, phi):
	N = r.size
	v = np.zeros((N,3))
	for i in range(N):
		v[i][0] = r[i] * np.sin(theta[i]) * np.cos(phi[i])
		v[i][1] = r[i] * np.sin(theta[i]) * np.sin(phi[i])
		v[i][2] = r[i] * np.cos(theta[i])
	return v