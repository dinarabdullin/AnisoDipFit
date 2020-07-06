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


def plot_confidence_intervals(parameters, score_vs_parameters, confidence_interval, numerical_error, best_parameters, save_figure=False, filename=''): 
    Np = sum(len(x) for x in parameters)
    c = 1
    fig = plt.figure(figsize=(18,9), facecolor='w', edgecolor='w')
    for i in range(len(parameters)):
        dim = len(parameters[i])
        for j in range(dim):
            plt.subplot(alignement[Np-1][0], alignement[Np-1][1], c)
            c = c + 1
            parameter_name = parameters[i][j]
            parameter_values = np.array([(x / const['variable_scales'][parameter_name]) for x in score_vs_parameters[i][parameter_name]])
            score_values = score_vs_parameters[i]['score']
            best_parameter = best_parameters[parameter_name]['value'] / const['variable_scales'][parameter_name]
            score_threshold = calculate_score_threshold(confidence_interval, numerical_error, 1)
            plot_confidence_interval(fig, parameter_values, score_values, parameter_name, best_parameter, score_threshold, numerical_error)
    ax_list = fig.axes
    handles, labels = ax_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.show(block=False)
    if save_figure:
        plt.savefig(filename, format='png', dpi=600)


def plot_confidence_interval(fig, parameter_values, score_values, parameter_name, best_parameter, score_threshold, numerical_error):
    # Determine the minimal and maximal values of a parameter
    parameter_min = np.amin(parameter_values)
    parameter_max = np.amax(parameter_values)
    # Create a new parameter axis
    Nx = 100
    parameter_axis = np.linspace(parameter_min, parameter_max, Nx)
    parameter_inc = parameter_axis[1] - parameter_axis[0]
    parameter_axis_lb = parameter_axis - parameter_inc * np.ones(parameter_axis.size)
    parameter_axis_lb[0] = parameter_axis[0]
    parameter_axis_ub = parameter_axis + parameter_inc * np.ones(parameter_axis.size)
    parameter_axis_ub[-1] = parameter_axis[-1]
    # Determine the minimal chi2 for each par value
    score_axis = np.ones(parameter_axis.size) * np.amax(score_values)
    for i in range(Nx):
        for j in range(parameter_values.size):
            if (parameter_axis_lb[i] < parameter_values[j]) and (parameter_values[j] < parameter_axis_ub[i]):
                if score_axis[i] > score_values[j]:
                    score_axis[i] = score_values[j] 
    # Set the optimal values of the parameter and score
    score_opt = np.amin(score_axis)
    idx_opt = np.argmin(score_axis)
    parameter_opt = parameter_axis[idx_opt]
    # Find the parameters ranges in which the score is within the confidence interval
    idx_ci = []
    for i in range(Nx):
        if (score_axis[i] - score_opt <= score_threshold):
            idx_ci.append(i)
    lower_bound = parameter_axis[idx_ci[0]]
    upper_bound = parameter_axis[idx_ci[-1]]
    # Plot the figure
    x = parameter_axis
    y = score_axis
    cmin = np.amin(score_values) + score_threshold
    cmax = 2 * cmin
    axes = fig.gca()
    axes.scatter(x, y, c=y, cmap='jet_r', vmin=cmin, vmax=cmax)
    axes.axvspan(lower_bound, upper_bound, facecolor="lightgray", alpha=0.3, label="confidence interval")
    x_best = best_parameter
    idx_best = min(range(len(x)), key=lambda i: abs(x[i]-x_best))
    y_best = y[idx_best]
    axes.plot(x_best, y_best, color='black', marker='o', markerfacecolor='white', markersize=12, clip_on=False, label = "fitting result")
    y1 = (np.amin(score_values) + score_threshold - numerical_error) * np.ones(x.size)
    y2 = (np.amin(score_values) + score_threshold) * np.ones(x.size)
    axes.plot(x, y1, 'm--', label = const['chi2_label']['conf_interval'])
    axes.plot(x, y2, 'k--', label = const['chi2_label']['conf_interval_inc_numerical_error'])
    axes.set_xlim(round(np.amin(parameter_values),1), round(np.amax(parameter_values),1))
    axes.set_xlabel(const['variable_labels'][parameter_name])
    axes.set_ylabel(const['chi2_label']['normalized_by_sn'])
    plt.margins(0.05)