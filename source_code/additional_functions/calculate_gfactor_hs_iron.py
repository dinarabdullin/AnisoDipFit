'''
calculate_gfactor_hs_iron.py

Calculates the g-factor of a high-spin Fe(III) based on its ZFS tensor
(only for positive ZFS, which is much larger than the Zeeman and thermal energies)

Requirements: Python3, numpy, scipy
'''

import sys
import numpy as np
from scipy import linalg
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)


const = {}
const['Hz2MHz'] = 1e-6
const['GHz2MHz'] = 1e3
const['wn2MHz'] = 29979.0
const['bohrMagneton'] = 9.274009994e-24 # J/T
const['plankConstant'] = 6.626070040e-34 # J*s
const['Fez'] = const['Hz2MHz'] * const['bohrMagneton'] / const['plankConstant'] # MHz/T


def spin_matrices(s):
	M = int(2*s + 1)
	SZ = np.zeros((M,M))
	SP = np.zeros((M,M))
	SM = np.zeros((M,M))
	SX = np.zeros((M,M))
	SY = np.zeros((M,M))
	SZ = np.diag(np.linspace(-s, s, M)) 
	for i in range(M-1):
		x = np.sqrt(float((i+1) * (M - (i+1))))
		SP[i][i+1] = x
		SM[i+1][i] = x
		SX[i][i+1] = 0.5 * x
		SX[i+1][i] = 0.5 * x
		SY[i][i+1] = -0.5 * (i+1) * x
		SY[i+1][i] = 0.5 * (i+1) * x
	return SZ, SP, SM, SX, SY


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


def gfactor_hs_iron(ga, zfs, fmw):
    # Actual g factors
    g_xx = ga[0]
    g_yy = ga[1]
    g_zz = ga[2]
    # g tensor in the molecular frame
    gv = np.array([g_xx, g_yy, g_zz])
    GT_M = np.diag(gv)
    # ZFS parameters
    D = zfs[0]
    E = zfs[1]
    # ZFS tensor in the molecular frame
    Dv = np.array([-D/3 + E, -D/3 - E, 2*D/3])
    DT_M = np.diag(Dv)
    # Electron spin operators
    SZ, SP, SM, SX, SY = spin_matrices(5/2)
    # Calculate effective g factor for 3 orientations along x,y,z
    orient = np.array([[0,         0.5*np.pi, 0],
                       [0.5*np.pi, 0.5*np.pi, 0],
                       [0,                 0, 0]])
    geff = np.zeros(3)
    for i in range(3):
        # Rotation matrix
        RM_L2M = euler2RM(orient[i])
        RM_M2L = RM_L2M.T
        # Electron Zeeman term of the Hamiltonian
        GT_L = np.dot(RM_L2M, np.dot(GT_M, RM_M2L))
        GT_L = 0.5 * (GT_L + GT_L.T)
        ez_M = np.array([0.0, 0.0, 1.0])
        ez_L = np.dot(GT_L, ez_M.T)
        Hez = const['Fez'] * ( (ez_L[0] - 1j*ez_L[1]) * 0.5 * SP + \
                               (ez_L[0] + 1j*ez_L[1]) * 0.5 * SM + \
                               ez_L[2] * SZ )
        # ZFS term of the Hamiltonian
        DT_L = np.dot(RM_L2M, np.dot(DT_M, RM_M2L))
        DT_L = 0.5 * (DT_L + DT_L.T)
        Hzfs = DT_L[2][2] * np.dot(SZ, SZ) + \
               0.25 * (DT_L[0][0] + DT_L[1][1]) * np.dot(SP, SM) + \
               0.25 * (DT_L[0][0] + DT_L[1][1]) * np.dot(SM, SP) + \
               0.25 * (DT_L[0][2] - 1j*DT_L[1][2]) * (np.dot(SZ, SP) + np.dot(SP, SZ)) + \
               0.25 * (DT_L[0][2] + 1j*DT_L[1][2]) * (np.dot(SZ, SM) + np.dot(SM, SZ)) + \
               0.25 * (DT_L[0][0] - DT_L[1][1] - 1j*(DT_L[0][1] + DT_L[1][0])) * np.dot(SP, SP) + \
               0.25 * (DT_L[0][0] - DT_L[1][1] + 1j*(DT_L[0][1] + DT_L[1][0])) * np.dot(SM, SM)
        # Calculate resonance fields via the eignenfield method
        E36 = np.eye(36)
        E6 = np.eye(6)
        A = const['GHz2MHz'] * fmw * E36 + np.kron(E6, Hzfs.T) - np.kron(Hzfs, E6)
        B = -np.kron(E6, Hez.T) + np.kron(Hez, E6)
        w, vr = linalg.eig(A, B)
        N = 0
        B0v = []
        for k in range(w.size):
            if (np.imag(w[k]) < 0.0001) & (np.real(w[k]) >= 0) & (np.real(w[k]) < 10):
                N += 1
                B0v.append(np.real(w[k]))
        # Find the resonance field of the lowest doublet
        B0 = 0
        dwMin = 0.1
        for k in range(N):
            Htot = B0v[k] * Hez + Hzfs
            w, vr = linalg.eig(Htot)
            w = sorted(np.real(w))
            dw = np.abs(w[0] - w[1]) - const['GHz2MHz'] * fmw
            if np.abs(dw) < dwMin:
                dwMin = np.abs(dw)
                B0 = B0v[k]
        # Calculate effective g factor
        geff[i] = const['GHz2MHz'] * fmw / (const['Fez'] * B0)
    return geff	


if __name__ == '__main__':
    # Input the values of g-factor
    var = input("\nEnter three principal values of a g-factor (default: 2 2 2):\n")
    if (var == ""):
        ga = np.array([2.0, 2.0, 2.0])
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 3:
            ga = np.array(val)
        else:
            raise ValueError('Could not obtain three values! Make sure that you use free spaces as delimiter.')
            sys.exit(1)
    # Input the value of ZFS in [cm-1]
    var = input("\nEnter axial and rhombic ZFS parameters in [cm-1] (default: 10 0):\n")
    if (var == ""):
        zfs = np.array([10.0, 0.0])
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 2:
            zfs = np.array(val)
        else:
            raise ValueError('Could not obtain two values! Make sure that you use free spaces as delimiter.')
            sys.exit(1)
    zfs = const['wn2MHz'] * zfs
    # Input the value of microwave frequency in [GHz]
    var = input("\nEnter the microwave frequency in [GHz] (default: 33.7):\n")
    if (var == ""):
        fmw = 33.7
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            fmw = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
    # Calculate g-factor
    geff = gfactor_hs_iron(ga, zfs, fmw)
    # Display g-factor
    sys.stdout.write("\nEffective g-factor: gxx = %f, gyy = %f, gzz = %f\n" % (geff[0], geff[1], geff[2]))
