'''
Saves the results of error analysis
'''

import sys
from fitting.output.save_fitting_parameters import save_fitting_parameters
from fitting.output.save_score_vs_parameters import save_score_vs_parameters


def save_error_analysis_data(optimizer, err_settings, output_settings):
    if output_settings['save_data'] and not (optimizer.score_vs_parameters == []):
        sys.stdout.write('Saving the results of the error analysis into the directory:\n')
        sys.stdout.write(output_settings['directory'])
        # Save the optimized fitting parameters
        filename = output_settings['directory'] + 'parameters.dat'
        save_fitting_parameters(optimizer.best_parameters, filename)
        # Save the score vs individual fitting parameters
        save_score_vs_parameters(err_settings['variables'], optimizer.score_vs_parameters, output_settings['directory'])
        # Output status
        sys.stdout.write(' [DONE]\n\n')  