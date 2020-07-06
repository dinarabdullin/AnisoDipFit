'''
Genetic Algorithm: Plot the results of the fitting
'''

import sys
import numpy as np
from fitting.graphics.plot_fit import plot_fit
from fitting.graphics.plot_score import plot_score


def plot_fitting_data(optimizer, exp_data, fit_settings, calc_settings, output_settings):
    sys.stdout.write('Plotting the fitting results... ')
    filename = output_settings['directory'] + 'fit.png'
    fig_fit, graph_fit = plot_fit(optimizer.best_fit, exp_data, fit_settings['settings']['fitted_data'], calc_settings, output_settings['save_figures'], filename)
    filename = output_settings['directory'] + 'score.png'
    fig_score, axes_score = plot_score(optimizer.best_score, True, output_settings['save_figures'], filename)
    sys.stdout.write('[DONE]\n\n')