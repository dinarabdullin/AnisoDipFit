'''
Plot the simulated dipolar spectra in dependence of the theta angle
'''

import numpy as np
import simulation.graphics.set_backend
import matplotlib.pyplot as plt
import simulation.graphics.set_style


def plot_spectrum_vs_theta(f_sim, spc_sim, theta, spc_vs_theta, normalized=False, xn=[], ranges=[], save_figure=False, filename=''):
    fig = plt.figure(figsize=(6,8), facecolor='w', edgecolor='w')
    axes = fig.add_subplot(211)
    if normalized:
        axes.plot(xn, spc_sim, 'k-')
        axes.set_xlim(np.amin(xn), np.amax(xn))
        axes.set_ylim(0, np.amax(spc_sim)+0.1)
        axes.set_xlabel(r'$\nu_{dd}$ ($\nu_{0}$)')
        axes.set_ylabel('Amplitude')
    else:
        axes.plot(f_sim, spc_sim, 'k-')
        if (ranges['f_max']):
            axes.set_xlim(-ranges['f_max'], ranges['f_max'])
            axes.set_ylim(0.0, ranges['spc_max']+0.1)
        else:
            axes.set_xlim(np.amin(f_sim), np.amax(f_sim))
            axes.set_ylim(0.0, np.amax(spc_sim)+0.1)
        axes.set_xlabel(r'Frequency (MHz)')
        axes.set_ylabel('Amplitude')
    axes = fig.add_subplot(212)
    if normalized:
        X, Y = np.meshgrid(xn, theta)
        Z = spc_vs_theta
        axes.contour(X, Y, Z, 100, cmap='jet', vmin=np.amin(Z), vmax=np.amax(Z))
        axes.set_xlim(np.amin(xn), np.amax(xn))
        axes.set_ylim(np.amin(theta), np.amax(theta))
        axes.set_xlabel(r'$\nu_{dd}$ ($\nu_{0}$)')
        axes.set_ylabel(r'$\mathit{\theta}$ (degree)')
    else:
        X, Y = np.meshgrid(f_sim, theta)
        Z = spc_vs_theta
        axes.contour(X, Y, Z, 100, cmap='jet', vmin=np.amin(Z), vmax=np.amax(Z))
        if (ranges['f_max']):
            axes.set_xlim(-ranges['f_max'], ranges['f_max'])
        else:
            axes.set_xlim(np.amin(f_sim), np.amax(f_sim))
        axes.set_ylim(np.amin(theta), np.amax(theta))
        axes.set_xlabel(r'Frequency (MHz)')
        axes.set_ylabel(r'$\mathit{\theta}$ (degree)')
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)