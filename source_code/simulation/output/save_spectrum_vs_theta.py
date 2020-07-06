'''
Save the simulated dipolar spectra in dependence of the angle theta
'''


def save_spectrum_vs_theta(f_sim, theta, spc_vs_theta, filename):
    file = open(filename, 'w')
    Nf = f_sim.size
    Ntheta = theta.size
    file.write('{0:<12s}'.format(' '))
    for k in range(Ntheta):
        file.write('{0:<12.4f}'.format(theta[k]))
    file.write('\n')
    for i in range(Nf):
        file.write('{0:<12.4f}'.format(f_sim[i]))
        for k in range(Ntheta):
            file.write('{0:<12.4f}'.format(spc_vs_theta[k][i]))
        file.write('\n')
    file.close()