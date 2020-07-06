'''
Genetic Algorithm: Plot fit to the experimental data
'''

import numpy as np
import fitting.graphics.set_backend
import matplotlib.pyplot as plt
import fitting.graphics.set_style


def plot_fit(fit, data, fitted_data, ranges, save_figure=False, filename=''):
    fig = plt.figure(facecolor='w', edgecolor='w')
    axes = fig.gca()
    if fitted_data == 'spectrum':
        axes.plot(data['f'], data['spc'], 'k-')
        graph = axes.plot(data['f'], fit, 'r--')
        axes.legend(('exp', 'fit'), loc='upper right', frameon=False)
        if ranges['f_max']:
            axes.set_xlim(-ranges['f_max'], ranges['f_max'])
        else:
            axes.set_xlim(np.amin(data['f']), np.amax(data['f']))
        axes.set_ylim(0.0, ranges['spc_max']+0.1)
        axes.set_xlabel(r'Frequency (MHz)')
        axes.set_ylabel('Amplitude')
    elif fitted_data == 'timetrace':
        axes.plot(data['t'], data['sig'], 'k-')
        graph = axes.plot(data['t'], fit, 'r--')
        axes.legend(('exp', 'fit'), loc='upper right', frameon=False)
        if ranges['t_max']:
            axes.set_xlim(ranges['t_min'], ranges['t_max'])
        else:
            axes.set_xlim(np.amin(data['t']), np.amax(data['t']))
        axes.set_ylim(np.amin(data['sig'])-0.2, np.amax(data['sig'])+0.1)
        axes.set_xlabel(r'$\mathit{t}$ ($\mathit{\mu s}$)')
        axes.set_ylabel('Echo intensity (a.u.)')   
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)
    return [fig, graph]

	
def update_fit_plot(fig, graph, fit):
	graph[0].set_ydata(fit)
	fig.canvas.draw()
	fig.canvas.flush_events()


def close_fit_plot(fig):
	plt.close(fig)