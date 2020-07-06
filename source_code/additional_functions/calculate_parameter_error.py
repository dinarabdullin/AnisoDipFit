'''
calculate_parameter_error.py 

Calculates the confidence interval(s) of the DipFit fitting parameter(s) par1 and par2

Requirements: Python3, wx, numpy, scipy, matplotlib 
'''

import os
import sys
import wx
import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.collections as collections
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
rcParams['font.size'] = 18

const = {}
const['variableLabels'] = {
	'r_mean'   : r'$\langle\mathit{r}\rangle$ (nm)',
	'r_width'  : r'$\mathit{\Delta r}$ (nm)', 
	'xi_mean'  : r'$\langle\mathit{\xi}\rangle$ $^\circ$', 
	'xi_width' : r'$\mathit{\Delta\xi}$ $^\circ$', 
	'phi_mean' : r'$\langle\mathit{\varphi}\rangle$ $^\circ$', 
	'phi_width': r'$\mathit{\Delta\varphi}$ $^\circ$', 
	'temp'     : r'Temperature (K)'}


def get_path(message):
    app = wx.App(None) 
    dialog = wx.FileDialog(None, message, wildcard='*.*', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    return path


def load_fit(filepath):
    x_data_list = []
    y_data_list = []
    y_fit_list = []
    file = open(filepath, 'r')
    for line in file:
        str = line.split()
        x_data_list.append(float(str[0]))
        y_data_list.append(float(str[1]))
        y_fit_list.append(float(str[2]))
    x_data_array = np.array(x_data_list)	
    y_data_array = np.array(y_data_list)	
    y_fit_array = np.array(y_fit_list)
    return [x_data_array, y_data_array, y_fit_list]    


def read_error_analysis_file(filepath):
    par1_list = []
    par2_list = []
    score_list = []
    file = open(filepath, 'r')
    next(file)
    for line in file:
        str = line.split()
        ncol = len(str)
        par1_list.append(float(str[0]))
        if ncol == 2:
            score_list.append(float(str[1]))
        elif ncol == 3:
            par2_list.append(float(str[1]))
            score_list.append(float(str[2]))
        else:
            print("Illegible data format!")
    par1_data = np.array(par1_list)
    par2_data = np.array(par2_list)
    score_data = np.array(score_list)  
    return [par1_data, par2_data, score_data]


def calculate_noise_std(y_data, y_fit):
    noise_std = 0
    noise_var = 0
    N = y_data.size
    norm = float(1/N)
    for i in range(N):
        noise_var = noise_var + (y_data[i]-y_fit[i])**2
    noise_std = np.sqrt(norm * noise_var)    
    return noise_std


def plot_noise(x_data, y_data, y_fit):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.plot(x_data, y_data, 'k-', label='exp')
    ax.plot(x_data, y_fit, 'r-', label='fit')
    ax.plot(x_data, y_data-y_fit, 'b-', label='noise')
    ax.set_xlim(np.amin(x_data), np.amax(x_data))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)
    

def rmsd2chi2(rmsd, noise_std, N):
    chi2 = rmsd**2 * N / noise_std**2
    return chi2


def drmsd2dchi2(rmsd, drmsd, noise_std, N):
    dchi2 = (drmsd**2 + 2*np.amin(rmsd)*drmsd) * N / noise_std**2
    return dchi2


def calculate_confidence_interval(par, chi2, par_name, par_opt, conf_int, calc_error):
    # Determine the minimal and maximal values of par
    par_min = np.amin(par)
    par_max = np.amax(par)
    
    # Create the new par-axis
    Nx = 101
    par_axis = np.linspace(par_min, par_max, Nx)
    par_inc = par_axis[1] - par_axis[0]
    par_axis_lb = par_axis - par_inc * np.ones(par_axis.size)
    par_axis_lb[0] = par_axis[0]
    par_axis_ub = par_axis + par_inc * np.ones(par_axis.size)
    par_axis_ub[-1] = par_axis[-1]
    
    # Determine the minimal chi2 for each par value
    chi2_axis = np.zeros(par_axis.size)
    for i in range(Nx):
        chi2_axis[i] = np.amin(chi2[(par_axis_lb[i] < par) & (par < par_axis_ub[i])])  
    
    # Determine the optimal values of par and chi2
    idx_opt = 0
    chi2_opt = 0
    if np.isnan(par_opt):
        chi2_opt = np.amin(chi2_axis)
        idx_opt = np.argmin(chi2_axis)
        par_opt = par_axis[idx_opt]
    else:
        idx_opt = min(range(len(par_axis.size)), key=lambda i: abs(par_axis[i]-par_opt))
        chi2_opt = chi2_axis[idx_opt]
    
    # Calculate (chi2 - chi2_opt) fo each par value
    chi2_diff = chi2_axis - chi2_opt*np.ones(Nx)
    
    # Find the ranges of var in which chi2_diff is within the confidence interval
    chi2_threshold = conf_int**2
    idx_error = []
    for i in range(Nx):
        if (chi2_diff[i] <= chi2_threshold):
            idx_error.append(i)
    par_left = par_axis[idx_error[0]]
    par_right = par_axis[idx_error[-1]]
    
    # Calculate the error of parameter
    par_error_left = np.abs(par_opt - par_left)
    par_error_right = np.abs(par_opt - par_right)
    par_error = np.nan
    if (par_error_left > par_min) or (par_error_right < par_max):
        if (par_error_left > par_error_right):
            par_error = par_error_left
        else:
            par_error = par_error_right  
    print('Error of %s: +/-%f (%d*sigma level)' % (par_name, par_error, conf_int))
    
    # Take into account the calculation error
    par_left2 = np.nan
    par_right2 = np.nan
    if not np.isnan(calc_error):
        chi2_diff2 = chi2_axis - chi2_opt*np.ones(Nx) - calc_error*np.ones(Nx)
        chi2_threshold = conf_int**2
        idx_error2 = []
        for i in range(Nx):
            if (chi2_diff2[i] <= chi2_threshold):
                idx_error2.append(i)
        par_left2 = par_axis[idx_error2[0]]
        par_right2 = par_axis[idx_error2[-1]]
        
        # Calculate the error of parameter
        par_error_left2 = np.abs(par_opt - par_left2)
        par_error_right2 = np.abs(par_opt - par_right2)
        par_error2 = np.nan
        if (par_error_left2 > par_min) or (par_error_right2 < par_max):
            if (par_error_left2 > par_error_right2):
                par_error2 = par_error_left2
            else:
                par_error2 = par_error_right2  
        print('Error of %s: +/-%f (%d*sigma level, chi2 calc. error = %f)' % (par_name, par_error2, conf_int, calc_error))
    
    # Plot results 
    par_ci = {}
    par_ci["ci"] = np.array([par_left, par_right])
    par_ci["ci&ce"] = np.array([par_left2, par_right2])
    plot_confidence_interval(par_axis, chi2_diff, par_ci, par_name, chi2_opt)
    
    return par_error


def plot_confidence_interval(par, chi2, par_ci, par_name, chi2_opt):    
    # Set the maximum and the minimum of chi2 for the colorbar
    chi2_min = chi2_opt
    chi2_max = 2 * chi2_min
    # Plot the figure
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.scatter(par, chi2, c=chi2, cmap='jet_r', vmin=chi2_min, vmax=chi2_max, label='data points')
    count = 0
    if not (np.isnan(par_ci["ci&ce"][0]) and np.isnan(par_ci["ci&ce"][1])):
        ax.axvspan(par_ci["ci&ce"][0], par_ci["ci&ce"][1], facecolor="orange", alpha=0.4, label="confidence interval")
    if not (np.isnan(par_ci["ci"][0]) and np.isnan(par_ci["ci"][1])):
        ax.axvspan(par_ci["ci"][0], par_ci["ci"][1], facecolor="green", alpha=0.2, label="confidence interval & calculation error")    
    ax.set_xlim(round(np.amin(par),1), round(np.amax(par),1))
    plt.xlabel(const['variableLabels'][par_name])
    plt.ylabel('chi2')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)


def keep_figures_live():
	plt.show()	


if __name__ == '__main__':
    # Load fit
    filepath = get_path("Open file with a fit")
    x_data, y_data, y_fit = load_fit(filepath)
    
    # Load error analysis data
    filepath = get_path("Open file with an error analysis")
    filedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath[:-4])
    filename_parts = filename.split("-")
    par1, par2, rmsd = read_error_analysis_file(filepath)
    
    # Input the confidence interval
    var = input("\nEnter the confidence interval in sigma units (default: 3): ")
    if (var == ""):
        conf_int = 3.0
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            conf_int = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
            
    # Input the optimal value of parameter 1
    var = input("\nEnter the optimal value of parameter 1 (default: NaN): ")
    if (var == ""):
        par1_opt = np.nan
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            par1_opt = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
    
    # Input the optimal value of parameter 2
    var = input("\nEnter the optimal value of parameter 2 (default: NaN): ")
    if (var == ""):
        par2_opt = np.nan
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            par2_opt = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)
    
    # Input the calculation error
    var = input("\nEnter the calculation error (default: 0.000275): ")
    if (var == ""):
        calc_error = 0.000275
    else:
        val = [float(i) for i in var.split(' ')]
        if len(val) == 1:
            calc_error = val[0]
        else:
            raise ValueError('More than one value obtained!')
            sys.exit(1)        

    # Estimate the standard deviation of noise
    noise_std = calculate_noise_std(y_data, y_fit)
    #plot_noise(x_data, y_data, y_fit)
    
    # Calculate the number of points
    N = y_data.size
    
    # Convert rmsd into chi2
    chi2 = rmsd2chi2(rmsd, noise_std, N)
    calc_error_chi2 = drmsd2dchi2(rmsd, calc_error, noise_std, N)
    
    # Calculate the confidence interval of parameter 1
    par1_name = filename_parts[1]
    par1_error = calculate_confidence_interval(par1, chi2, par1_name, par1_opt, conf_int, calc_error_chi2)
    
    # Calculate the confidence interval of parameter 2
    if not (par2.size == 0):
        par2_name = filename_parts[2]
        par2_error = calculate_confidence_interval(par2, chi2, par2_name, par2_opt, conf_int, calc_error_chi2)
    
    print('Done!')
    keep_figures_live()    