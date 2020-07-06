'''
Genetic Algorithm: Plot the score in dependence of individual parameters
'''

import sys
import numpy as np
import scipy
from supplement.constants import const
import fitting.graphics.set_backend
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import fitting.graphics.set_style
from supplement.constants import const
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from fitting.error_estimation import calculate_score_threshold

# No. of plots: 1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16
alignement = [[1,1], [1,2], [1,3], [2,2], [2,3], [2,3], [2,4], [2,4], [3,3], [3,4], [3,4], [3,4], [4,4], [4,4], [4,4], [4,4]]


def plot_score_vs_parameters(parameters, score_vs_parameters, confidence_interval, numerical_error, best_parameters, save_figure=False, filename=''): 
    Ne = len(parameters)
    c = 1
    fig = plt.figure(figsize=(18,9), facecolor='w', edgecolor='w')
    for i in range(Ne):
        dim = len(parameters[i])
        score_threshold = calculate_score_threshold(confidence_interval, numerical_error, dim)
        if (dim == 1):
            plt.subplot(alignement[Ne-1][0], alignement[Ne-1][1], c)
            c = c + 1
            im = plot_1d(fig, parameters[i], score_vs_parameters[i], score_threshold, best_parameters)
        elif (dim == 2):
            plt.subplot(alignement[Ne-1][0], alignement[Ne-1][1], c)
            c = c + 1
            im = plot_2d(fig, parameters[i], score_vs_parameters[i], score_threshold, best_parameters)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.80, top=0.9)
    cax = plt.axes([0.85, 0.3, 0.02, 0.4]) # left, bottom, width, height  
    plt.colorbar(im, cax=cax, orientation='vertical')
    plt.text(0.90, 1.05, const['chi2_label']['normalized_by_sn'], transform=cax.transAxes)
    plt.draw()
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)


def plot_1d(fig, parameters, score_vs_parameters, score_threshold, best_parameters):
    # Read out the values of fitting parameters and RMSD values
    name1 = parameters[0]
    x1 = [(x / const['variable_scales'][name1]) for x in score_vs_parameters[name1]]
    y = score_vs_parameters['score']
    # Read out the optimized values of fitting parameters
    x1_opt = best_parameters[name1]['value'] / const['variable_scales'][name1]
    y_opt = np.amin(y)
    # Set the maximum and the minimum of y
    cmin = np.amin(y) + score_threshold
    cmax = 2 * cmin
    # Plot the figure
    axes = fig.gca()
    im = axes.scatter(x1, y, c=y, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.set_xlim(round(np.amin(x1),1), round(np.amax(x1),1))
    axes.set_xlabel(const['variable_labels'][name1])
    axes.set_ylabel(const['chi2_label']['normalized_by_sn'])
    axes.plot(x1_opt, y_opt, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False)
    plt.margins(0.05)
    # Make the axes of equal scale
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect( (xh-xl)/(yh-yl) )
    return im


def plot_2d(fig, parameters, score_vs_parameters, score_threshold, best_parameters):
    # Read out the values of fitting parameters and RMSD values
    name1 = parameters[0]
    name2 = parameters[1]
    x1 = [(x / const['variable_scales'][name1]) for x in score_vs_parameters[name1]]	
    x2 = [(x / const['variable_scales'][name2]) for x in score_vs_parameters[name2]]	
    y = score_vs_parameters['score']
    # Read out th optimized values of fitting parameters
    x1_opt = best_parameters[name1]['value'] / const['variable_scales'][name1]
    x2_opt = best_parameters[name2]['value'] / const['variable_scales'][name2]
    # Interpolate the data on a regular grid
    x1min = np.min(x1)
    x1max = np.max(x1)
    x2min = np.min(x2)
    x2max = np.max(x2)
    x1r = np.linspace(x1min, x1max, num=200) 
    x2r = np.linspace(x2min, x2max, num=200)
    X, Y = np.mgrid[x1min:x1max:200j, x2min:x2max:200j]
    Z = griddata((x1, x2), y, (X, Y), method='linear')
    # Set the maximum and the minimum of Z
    cmin = np.amin(y) + score_threshold
    cmax = 2 * cmin
    # Plot the figure
    axes = fig.gca()
    im = axes.pcolor(X, Y, Z, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.set_xlim(np.amin(x1), np.amax(x1))
    axes.set_ylim(np.amin(x2), np.amax(x2))
    axes.set_xlabel(const['variable_labels'][name1])
    axes.set_ylabel(const['variable_labels'][name2])
    axes.plot(x1_opt, x2_opt, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False)
    plt.margins(0.05)
    # Make the axis of equal scale
    xl, xh = axes.get_xlim()
    yl, yh = axes.get_ylim()
    axes.set_aspect( (xh-xl)/(yh-yl) )
    # Make ticks
    plt.xticks(np.linspace(round(x1min,1), round(x1max,1), 3))
    plt.yticks(np.linspace(round(x2min,1), round(x2max,1), 3))
    axes.xaxis.labelpad = 0
    axes.yaxis.labelpad = 0
    return im