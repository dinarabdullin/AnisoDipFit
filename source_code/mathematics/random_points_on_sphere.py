'''
Generate random points on a sphere
'''

import math
import numpy as np
from mathematics.spherical2cartesian import spherical2cartesian


def random_points_on_sphere(N):
	r = np.ones(N)
	theta = np.arccos(2.0 * np.random.random_sample(N) - 1.0)
	phi = 2.0 * np.pi * np.random.random_sample(N)
	v = spherical2cartesian(r, theta, phi)
	return v