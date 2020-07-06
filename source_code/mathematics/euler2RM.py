'''
Convert ZYZ Euler angles into a rotation matrix
'''

import numpy as np


def euler2RM(euler):
	alpha = euler[0]
	beta = euler[1]
	gamma = euler[2]
	sa = np.sin(alpha)
	ca = np.cos(alpha)
	sb = np.sin(beta)
	cb = np.cos(beta)
	sg = np.sin(gamma)
	cg = np.cos(gamma)
	RM = np.zeros((3,3))
	RM[0][0] = cg*cb*ca - sg*sa
	RM[0][1] = cg*cb*sa + sg*ca
	RM[0][2] = -cg*sb
	RM[1][0] = -sg*cb*ca - cg*sa
	RM[1][1] = -sg*cb*sa + cg*ca
	RM[1][2] = sg*sb
	RM[2][0] = sb*ca
	RM[2][1] = sb*sa
	RM[2][2] = cb
	return RM