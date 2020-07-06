'''
Plot the flip probability of the B-spin vs g-factor and temperature
'''

import numpy as np
import simulation.graphics.set_backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import simulation.graphics.set_style


def plot_pb_vs_temp(g, temp, pb_vs_temp, save_figure=False, filename=''): 
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    Ntemp = len(temp)
    colors = cm.rainbow(np.linspace(0, 1, Ntemp))
    for i in range(Ntemp):
        axes.plot(g, pb_vs_temp[i], color=colors[i])
    axes.set_xlim(np.amin(g), np.amax(g))
    axes.set_ylim(np.amin(pb_vs_temp)-0.1, np.amax(pb_vs_temp)+0.1)
    axes.set_xlabel(r'$\mathit{g_{eff}}$')
    axes.set_ylabel('Modulation depth')
    labels = [(str(t) + ' K') for t in temp]
    axes.legend(labels=labels, loc='lower left')
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)