'''
Plot the simulated and experimental dipolar spectra
'''

import numpy as np
import simulation.graphics.set_backend
import matplotlib.pyplot as plt
import simulation.graphics.set_style


def plot_spectrum(f_sim, spc_sim, f_exp, spc_exp, normalized=False, xn=[], ranges=[], save_figure=False, filename=''):
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    if not (f_exp == []):
        axes.plot(f_exp, spc_exp, 'k-')
        axes.plot(f_sim, spc_sim, 'r--')
        axes.legend(('exp', 'fit'), loc='upper right', frameon=False)
        if not (ranges == []):
            axes.set_xlim(-ranges['f_max'], ranges['f_max'])
            axes.set_ylim(0.0, ranges['spc_max']+0.1)
        else:
            axes.set_xlim(np.amin(f_exp), np.amax(f_exp))
            axes.set_ylim(0.0, np.amax(spc_exp)+0.1)
        axes.set_xlabel(r'Frequency (MHz)')
        axes.set_ylabel('Amplitude')
    else:
        if normalized:
            axes.plot(xn, spc_sim, 'k-')
            axes.set_xlim(np.amin(xn), np.amax(xn))
            axes.set_ylim(0, np.amax(spc_sim)+0.1)
            axes.set_xlabel(r'$\nu_{dd}$ ($\nu_{0}$)')
            axes.set_ylabel('Amplitude')
        else:
            axes.plot(f_sim, spc_sim, 'k-')
            axes.set_xlim(np.amin(f_sim), np.amax(f_sim))
            axes.set_ylim(0.0, np.amax(spc_sim)+0.1)
            axes.set_xlabel(r'Frequency (MHz)')
            axes.set_ylabel('Amplitude')
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)