'''
display_error_analysis_data.py 

Plots the error analysis data of AnisoDipFit

Requirements: Python3, wx, numpy, scipy, matplotlib 
'''

import os
import io
import sys
import wx
import numpy as np
from copy import deepcopy
import scipy
import scipy.stats
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['axes.facecolor']= 'white'
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['lines.linewidth'] = 2
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.5


const = {}
const['variable_names'] = [
	'r_mean',
	'r_width', 
	'xi_mean', 
	'xi_width', 
	'phi_mean', 
	'phi_width', 
	'temp']
const['long_variable_names'] = {
	'r_mean'   : 'r mean (nm)',
	'r_width'  : 'r width (nm)', 
	'xi_mean'  : 'xi mean (deg)',
	'xi_width' : 'xi width (deg)', 
	'phi_mean' : 'phi mean (deg)', 
	'phi_width': 'phi mean (deg)',
	'temp'     : 'temperature (K)'}
const['variable_scales'] = {
	'r_mean'   : 1.0,
	'r_width'  : 1.0, 
	'xi_mean'  : np.pi / 180.0, 
	'xi_width' : np.pi / 180.0, 
	'phi_mean' : np.pi / 180.0, 
	'phi_width': np.pi / 180.0, 
	'temp'     : 1.0}
const['variable_labels'] = {
	'r_mean'   : r'$\langle\mathit{r}\rangle$ (nm)',
	'r_width'  : r'$\mathit{\Delta r}$ (nm)', 
	'xi_mean'  : r'$\langle\mathit{\xi}\rangle$ $^\circ$', 
	'xi_width' : r'$\mathit{\Delta\xi}$ $^\circ$', 
	'phi_mean' : r'$\langle\mathit{\varphi}\rangle$ $^\circ$', 
	'phi_width': r'$\mathit{\Delta\varphi}$ $^\circ$', 
	'temp'     : r'Temperature (K)'}
const['chi2_label'] = {
    'unitary_sn'                        : r'$\mathit{\chi^2}$ ($\mathit{\sigma_{N}}$ = 1)',
    'normalized_by_sn'                  : r'$\mathit{\chi^2}$',
    'conf_interval'                     : r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}_{ci}}$',
    'conf_interval_inc_numerical_error' : r'$\mathit{\chi^{2}_{min}}$ + $\mathit{\Delta\chi^{2}_{ci}}$ + $\mathit{\Delta\chi^{2}_{ne}}$'}


error_analysis_files = [
    {'filename': 'parameter_errors-r_mean-r_width.dat',     '2d': True, 'x1': 'r_mean',     'x2': 'r_width'     },
    {'filename': 'parameter_errors-xi_mean-xi_width.dat',   '2d': True, 'x1': 'xi_mean',    'x2': 'xi_width'    },
    {'filename': 'parameter_errors-phi_mean-phi_width.dat', '2d': True, 'x1': 'phi_mean',   'x2': 'phi_width'   },
    {'filename': 'parameter_errors-xi_mean-phi_mean.dat',   '2d': True, 'x1': 'xi_mean',    'x2': 'phi_mean'    },
    {'filename': 'parameter_errors-r_width-xi_mean.dat',    '2d': True, 'x1': 'r_width',    'x2': 'xi_mean'     },
    {'filename': 'parameter_errors-r_width-xi_width.dat',   '2d': True, 'x1': 'r_width',    'x2': 'xi_width'    },
    {'filename': 'parameter_errors-r_width-phi_mean.dat',   '2d': True, 'x1': 'r_width',    'x2': 'phi_mean'    },
    {'filename': 'parameter_errors-r_width-phi_width.dat',  '2d': True, 'x1': 'r_width',    'x2': 'phi_width'   },
    {'filename': 'parameter_errors-r_mean.dat',             '2d': False,'x1': 'r_mean',     'x2': ''            },
    {'filename': 'parameter_errors-r_width.dat',            '2d': False,'x1': 'r_width',    'x2': ''            },
    {'filename': 'parameter_errors-xi_mean.dat',            '2d': False,'x1': 'xi_mean',    'x2': ''            },
    {'filename': 'parameter_errors-xi_width.dat',           '2d': False,'x1': 'xi_width',   'x2': ''            },
    {'filename': 'parameter_errors-phi_mean.dat',           '2d': False,'x1': 'phi_mean',   'x2': ''            },
    {'filename': 'parameter_errors-phi_width.dat',          '2d': False,'x1': 'phi_width',  'x2': ''            },
    {'filename': 'parameter_errors-temp.dat',               '2d': False,'x1': 'temp',       'x2': ''            },
]


# No. of plots: 1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16
alignement = [[1,1], [1,2], [1,3], [2,2], [2,3], [2,3], [2,4], [2,4], [3,3], [3,4], [3,4], [3,4], [4,4], [4,4], [4,4], [4,4]]


def get_path(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.*', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    return path


def load_error_analysis_data(directory):
    nfiles = len(error_analysis_files)
    parameters = []
    score_vs_parametes = []
    for i in range(nfiles):
        filename = directory + error_analysis_files[i]['filename']
        # check if the file exists
        if os.path.isfile(filename):
            # set parameters
            if error_analysis_files[i]['2d'] == False:
                parameters_from_one_file = [error_analysis_files[i]['x1']]
            else:
                parameters_from_one_file = [error_analysis_files[i]['x1'], error_analysis_files[i]['x2']]
            parameters.append(parameters_from_one_file)
            # read data from the file
            data = np.genfromtxt(filename, skip_header=1)
            nx = data.shape[0]
            score_vs_parameters_from_one_file = {}
            if error_analysis_files[i]['2d'] == False:
                name1 = error_analysis_files[i]['x1']
                score_vs_parameters_from_one_file[name1] = [data[j][0] * const['variable_scales'][name1] for j in range(nx)]
                score_vs_parameters_from_one_file['score'] = [data[j][1] for j in range(nx)]
            else:
                name1 = error_analysis_files[i]['x1']
                name2 = error_analysis_files[i]['x2']
                score_vs_parameters_from_one_file[name1] = [data[j][0] * const['variable_scales'][name1] for j in range(nx)]
                score_vs_parameters_from_one_file[name2] = [data[j][1] * const['variable_scales'][name2] for j in range(nx)]
                score_vs_parameters_from_one_file['score'] = [data[j][2] for j in range(nx)]
            score_vs_parametes.append(score_vs_parameters_from_one_file)
    return [parameters, score_vs_parametes]


def load_optimized_parameters(filepath):
    optimized_parameters = {}
    count = 0
    file = open(filepath, 'r')
    # Skip a header
    next(file)
    for line in file:
        str = list(chunkstring(line, 16))
        parameter = {}
        name = const['variable_names'][count] 
        parameter['longname'] = str[0].strip()
        parameter['value'] = float(str[1]) * const['variable_scales'][name]
        parameter['optimized'] = str[2].strip()
        parameter['precision'] = float(str[3]) * const['variable_scales'][name]
        optimized_parameters[name] = parameter
        count += 1
    return optimized_parameters


def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


def calculate_score_threshold(confidence_interval, numerical_error, degree_of_freedom):
    score_threshold = 0.0
    if degree_of_freedom == 1:
        score_threshold = confidence_interval**2 + numerical_error
    else:
        p = 1.0 - scipy.stats.chi2.sf(confidence_interval**2, 1)
        score_threshold = scipy.stats.chi2.ppf(p, int(degree_of_freedom)) + numerical_error
    return score_threshold


def calculate_parameter_error(parameter_values, score_values, score_threshold):
    # Determine the minimal and maximal values of the parameter
    parameter_min = np.amin(parameter_values)
    parameter_max = np.amax(parameter_values)
    # Determine the minimal score value and the corresponding parameter value
    score_opt = np.amin(score_values)
    idx_opt = np.argmin(score_values)
    parameter_opt = parameter_values[idx_opt]
    # Determine the parameter values which lie under the score threshold 
    parameter_selected = []
    for i in range(parameter_values.size):
        score_diff = score_values[i] - score_opt
        if score_diff <= score_threshold:
            parameter_selected.append(parameter_values[i])
    # Determine the uncertainty ranges of the parameter
    lower_bound = np.amin(np.array(parameter_selected))
    upper_bound = np.amax(np.array(parameter_selected))
    parameter_error_lb = np.abs(parameter_opt - lower_bound)
    parameter_error_ub = np.abs(parameter_opt - upper_bound)
    parameter_error = np.nan
    if (lower_bound > parameter_min) or (upper_bound < parameter_max):
        if (parameter_error_lb > parameter_error_ub):
            parameter_error = parameter_error_lb
        else:
            parameter_error = parameter_error_ub  
    return parameter_error


def include_errors(optimized_parameters, parameter_errors):
    optimized_parameters_with_errors = deepcopy(optimized_parameters)
    for name in parameter_errors:
        if not np.isnan(parameter_errors[name]):
            optimized_parameters_with_errors[name]['precision'] = parameter_errors[name]
    return optimized_parameters_with_errors


def print_optimized_parameters(optimized_parameters):
        sys.stdout.write('Optimized fitting parameters:\n')
        sys.stdout.write("{0:<16s} {1:<16s} {2:<16s} {3:<16s}\n".format('Parameter', 'Value', 'Optimized', 'Precision (+/-)'))
        for name in const['variable_names']:
            parameter = optimized_parameters[name]
            sys.stdout.write("{0:<16s} ".format(parameter['longname']))
            sys.stdout.write("{0:<16.3f} ".format(parameter['value'] / const['variable_scales'][name]))
            sys.stdout.write("{0:<16s} ".format(parameter['optimized']))
            sys.stdout.write("{0:<16.3f} \n".format(parameter['precision'] / const['variable_scales'][name]))
        sys.stdout.write('\n')


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


def keep_figures_live():
	plt.show()	

	
if __name__ == '__main__':
    # Read the results of the error analysis
    filepath = get_path("Open file with the results of error analysis...")
    directory = os.path.dirname(filepath) + '/'
    parameters, score_vs_parameters = load_error_analysis_data(directory)
    
    # Read the optimized fitting parameters
    filepath2 = get_path("Open file with optimized fitting parameters...")
    best_parameters = load_optimized_parameters(filepath2)
    
    # Input the confidence interval
    var = input("\nEnter the confidence interval in sigma units (default: 3): ")
    if (var == ""):
        confidence_interval = 3.0
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            confidence_interval = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
    
    # Input the numerical error
    var = input("\nEnter the numerical error in chi2 units (default: 0): ")
    if (var == ""):
        numerical_error = 0.0
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            numerical_error = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
    
    # Input the font size
    var = input("\nEnter the font size (default: 28): ")
    if (var == ""):
        fontsize = 28
    else:
        val = [int(i) for i in var.split(' ')]
        if len(val) == 1:
            fontsize = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
    matplotlib.rcParams.update({'font.size': fontsize}) 
    
    # Calculate parameters' errors
    parameter_errors = {}
    Ne = len(score_vs_parameters)
    for i in range(Ne):
        for name in parameters[i]:
            parameter_values = np.array(score_vs_parameters[i][name])
            score_values = np.array(score_vs_parameters[i]['score'])
            score_threshold = calculate_score_threshold(confidence_interval, numerical_error, 1)
            parameter_error = calculate_parameter_error(parameter_values, score_values, score_threshold)
            if name in parameter_errors:
                if not np.isnan(parameter_error) and not np.isnan(parameter_errors[name]):
                    if (parameter_error > parameter_errors[name]):
                        parameter_errors[name] = parameter_error
            else:
                parameter_errors[name] = parameter_error
    best_parameters_with_errors = include_errors(best_parameters, parameter_errors)
    
    # Display the optimized fitting parameters with errors
    print_optimized_parameters(best_parameters_with_errors)
    
    # Plot the error analysis data
    filename = directory + 'parameter_errors_formated.png'
    plot_score_vs_parameters(parameters, score_vs_parameters, confidence_interval, numerical_error, best_parameters, True, filename)	
    keep_figures_live()