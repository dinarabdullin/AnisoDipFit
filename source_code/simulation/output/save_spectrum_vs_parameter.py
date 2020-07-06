'''
Save the simulated dipolar spectra in dependence of some parameter
'''


def save_spectrum_vs_parameter(f_sim, parameter, spc_vs_parameter, filename):
    file = open(filename, 'w')
    Nf = f_sim.size
    Np = parameter.size
    file.write('{0:<12s}'.format(' '))
    for k in range(Np):
        file.write('{0:<12.4f}'.format(parameter[k]))
    file.write('\n')
    for i in range(Nf):
        file.write('{0:<12.4f}'.format(f_sim[i]))
        for k in range(Np):
            file.write('{0:<12.4f}'.format(spc_vs_parameter[k][i]))
        file.write('\n')
    file.close()