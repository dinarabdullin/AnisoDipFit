'''
Plot the simulated dipolar spectra in dependence of some parameter
'''

import numpy as np
import simulation.graphics.set_backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import simulation.graphics.set_style


def plot_spectrum_vs_parameter_3d(f_sim, parameter, spc_vs_parameter, normalized=False, xn=[], ranges=[], 
                                  save_figure=False, filename='', par_label='', invert_parameter_axis=False): 
    # Colormap
    Np = parameter.size
    Nc = int(1.5 * Np)
    cmap = plt.cm.get_cmap('gray', Nc)
    cmaplist = [cmap(i) for i in range(Nc)]
    # Plot the spectrum vs parameter
    fig = plt.figure(figsize=(9,9), facecolor='w', edgecolor='w')
    axes = fig.gca(projection='3d') 
    if normalized:
        for i in range(Np):
            axes.plot(xn, parameter[i]*np.ones(f_sim.size), spc_vs_parameter[i], color=cmaplist[i])
        axes.set_xlim(np.amin(xn), np.amax(xn))
        axes.set_ylim(np.amin(parameter), np.amax(parameter))
        axes.set_zlim(0.0, np.amax(spc_vs_parameter)+0.1)
        axes.set_xlabel(r'$\nu_{dd}$ ($\nu_{0}$)', labelpad=20)
        axes.set_ylabel(par_label, labelpad=20)
        axes.set_zlabel('Amplitude', labelpad=20)
    else:
        for i in range(Np):
            if ranges['f_max']:
                idx = [j for j in range(f_sim.size) if (f_sim[j] >= -ranges['f_max']) and (f_sim[j] <= ranges['f_max'])]
                f_selected = np.array([f_sim[j] for j in idx])
                spc_vs_parameter_selected = np.array([spc_vs_parameter[i][j] for j in idx])
                axes.plot(f_selected, parameter[i]*np.ones(f_selected.size), spc_vs_parameter_selected, color=cmaplist[i])
                axes.set_xlim(np.amin(f_selected), np.amax(f_selected))
            else:
                axes.plot(f_sim, parameter[i]*np.ones(f_sim.size), spc_vs_parameter[i], color=cmaplist[i])
                axes.set_xlim(np.amin(f_sim), np.amax(f_sim))
        axes.set_ylim(np.amin(parameter), np.amax(parameter))      
        axes.set_zlim(0.0, np.amax(spc_vs_parameter)+0.1)
        axes.set_xlabel(r'Frequency (MHz)', labelpad=20)
        axes.set_ylabel(par_label, labelpad=20)
        axes.set_zlabel('Amplitude', labelpad=20)
    axes.tick_params(axis='y', which='major', pad=10)
    axes.tick_params(axis='z', which='major', pad=10)
    axes.view_init(elev=45, azim=-85)
    if invert_parameter_axis:
        axes.invert_yaxis()
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)