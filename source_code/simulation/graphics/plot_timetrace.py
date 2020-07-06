'''
Plot the simulated and experimental PDS time traces
'''

import numpy as np
import simulation.graphics.set_backend
import matplotlib.pyplot as plt
import simulation.graphics.set_style


def plot_timetrace(t_sim, sig_sim, t_exp, sig_exp, save_figure=False, filename=''):
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    if not (t_exp == []):    
        axes.plot(t_exp, sig_exp, 'k-')
        axes.plot(t_sim, sig_sim, 'r--')	
        axes.legend(('exp', 'fit'), loc='upper right', frameon=False)	
    else:
        axes.plot(t_sim, sig_sim, 'r-')	
    plt.xlim([min(t_sim), max(t_sim)])
    plt.ylim([np.amin(sig_sim)-0.1, 1.1])
    plt.xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
    plt.ylabel('Echo intensity (a.u.)')
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)