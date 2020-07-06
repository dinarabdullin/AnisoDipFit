'''
Plot the simulated dipolar spectra in dependence of some parameter
'''

import numpy as np
import math
import simulation.graphics.set_backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import simulation.graphics.set_style


def plot_spectrum_vs_parameter_2d(f_sim, parameter, spc_vs_parameter, normalized=False, xn=[], ranges=[], 
                                  save_figure=False, filename='', par_label='', invert_parameter_axis=False):
    fig = plt.figure(figsize=(6,10), facecolor='w', edgecolor='w')
    fig.subplots_adjust(left=.12, right=.85, bottom=.10, top=.95)
    axes = fig.gca()
    Nx = f_sim.size
    Np = parameter.size
    Nskip = Nskip = int((Np-1) / 10) + 1
    count = 0
    # Set the values of increment and baseline
    if invert_parameter_axis:
        increment = -0.7 * np.amax(spc_vs_parameter)
        baseline = (-increment * Np) * np.ones(Nx)
    else:
        increment = 0.7 * np.amax(spc_vs_parameter)
        baseline = np.zeros(Nx)
    if normalized:
        for i in range(Np):
            axes.plot(xn, spc_vs_parameter[i] + baseline, 'k-')          
            if (count == 0):
                plt.text(np.amax(xn)+0.5, baseline[0]-0.07, str(int(parameter[i])))
            count = count + 1
            if (count == Nskip):
                count = 0
            baseline = baseline + increment * np.ones(Nx)
            if (i == Np-1):
                plt.text(np.amax(xn)+0.5, baseline[0]+0.4*Nskip*increment, par_label)
        axes.set_xlim(np.amin(xn), np.amax(xn))
        axes.set_xlabel(r'$\nu_{dd}$ ($\nu_{0}$)')
        axes.set_xticks(np.linspace(-math.floor(np.amax(xn)), math.floor(np.amax(xn)), 2 * int(math.floor(np.amax(xn))) + 1))
        axes.xaxis.grid(color='darkgray', linestyle='--', linewidth=1)
    else:
        for i in range(Np):
            axes.plot(f_sim, spc_vs_parameter[i]+baseline, 'k-')
            if (count == 0):
                plt.text(np.amax(f_sim)+0.5, baseline[0]-0.07, str(int(parameter[i])))
            count = count + 1
            if (count == Nskip):
                count = 0
            baseline = baseline + increment * np.ones(Nx)
            if (i == Np-1):   
                plt.text(np.amax(f_sim)+0.5, baseline[0]+0.4*Nskip*increment, par_label)
        axes.set_xlim(np.amin(f_sim), np.amax(f_sim))
        axes.set_xlabel(r'Frequency (MHz)')
    #axes.set_ylabel('Amplitude')   
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False, labelright = False)
    axes.tick_params(axis = 'x', which = 'both', top = False, bottom = True)
    #plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)