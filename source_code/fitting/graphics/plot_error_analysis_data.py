'''
Genetic Algorithm: Plot the results of the fitting
'''

import sys
import numpy as np
from fitting.graphics.plot_score_vs_parameters import plot_score_vs_parameters
from fitting.graphics.plot_confidence_intervals import plot_confidence_intervals


def plot_error_analysis_data(optimizer, err_settings, fit_settings, output_settings):
    if not (optimizer.score_vs_parameters == []):
        sys.stdout.write('Plotting the results of the error analysis... ')
        filename = output_settings['directory'] + 'parameter_errors.png'
        plot_score_vs_parameters(err_settings['variables'], optimizer.score_vs_parameters, err_settings['confidence_interval'], optimizer.numerical_error, optimizer.best_parameters, output_settings['save_figures'], filename)
        filename = output_settings['directory'] + 'confidence_intervals.png'
        plot_confidence_intervals(err_settings['variables'], optimizer.score_vs_parameters, err_settings['confidence_interval'], optimizer.numerical_error, optimizer.best_parameters, output_settings['save_figures'], filename)
        sys.stdout.write('[DONE]\n\n')