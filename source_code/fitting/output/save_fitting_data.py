'''
Save fitting results
'''

import sys
from fitting.output.save_fit import save_fit
from fitting.output.save_score import save_score
from fitting.output.save_fitting_parameters import save_fitting_parameters


def save_fitting_data(optimizer, exp_data, fit_settings, output_settings):
    if output_settings['save_data']:
        sys.stdout.write('Saving the fitting results into the directory:\n')
        sys.stdout.write(output_settings['directory'])
        # Save the fit
        filename = output_settings['directory'] + 'fit.dat'
        save_fit(optimizer.best_fit, exp_data, fit_settings['settings']['fitted_data'], filename)
        # Saves the score vs optimization step
        filename = output_settings['directory'] + 'score.dat'
        save_score(optimizer.best_score, filename)
        # Save the optimized fitting parameters
        filename = output_settings['directory'] + 'parameters.dat'
        save_fitting_parameters(optimizer.best_parameters, filename)
        sys.stdout.write(' [DONE]\n\n')  