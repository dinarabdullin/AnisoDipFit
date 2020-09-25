'''
Estimate the error of the optimized fitting parameters
'''

import sys
import numpy as np
import scipy
import scipy.stats
from scipy.optimize import curve_fit
from functools import partial
from fitting.generation import Generation
from supplement.constants import const


def calculate_score_vs_parameters(best_parameters, err_settings, fit_settings, simulator, exp_data, spinA, spinB, calc_settings):
    sys.stdout.write("Calculating chi2 in dependence of fitting parameters ...\n")
    score_vs_parameters = []
    Ne = len(err_settings['variables'])
    Ns = err_settings['Ns']
    for i in range(Ne):
        sys.stdout.write('\r')
        sys.stdout.write("Calculation step %d / %d" % (i+1, Ne))
        sys.stdout.flush()
        # Create a generation
        generation = Generation(Ns)
        generation.first_generation(fit_settings['parameters']['bounds'])
        # Set all genes to the optimized values except the ones which will to be varied  
        for name in const['variable_names']:
            index = fit_settings['parameters']['indices'][name]
            if not (index == -1) and not (name in err_settings['variables'][i]):
                for j in range(Ns):
                    generation.chromosomes[j].genes[index] = best_parameters[name]['value']
        # Score the generation
        generation.score_chromosomes(fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
        # Sort chromosomes according to their score
        generation.sort_chromosomes()
        # Store the variables and the corresponding score values
        score_vs_variables = {}
        for name in err_settings['variables'][i]:
            score_vs_variables[name] = []
            index = fit_settings['parameters']['indices'][name]
            for j in range(Ns):
                score_vs_variables[name].append(generation.chromosomes[j].genes[index])
        score_vs_variables['score'] = []
        for j in range(Ns):
            score_vs_variables['score'].append(generation.chromosomes[j].score)
        score_vs_parameters.append(score_vs_variables)
    sys.stdout.write('\n')
    return score_vs_parameters


def calculate_parameter_errors(score_vs_parameters, best_parameters, err_settings, fit_settings, simulator, exp_data, spinA, spinB, calc_settings):
    sys.stdout.write("Calculating the errors of fitting parameters ...\n")
    # Determine the calculation error
    numerical_error = calculate_numerical_error(best_parameters, err_settings, fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
    sys.stdout.write("Numerical error (chi2) = %f\n" % (numerical_error))
    # Set the score threshold
    score_threshold = calculate_score_threshold(err_settings['confidence_interval'], numerical_error, 1)
    sys.stdout.write("Score threshold (chi2) = %f\n" % (score_threshold))
    # Calculate the errors of fitting parameters
    parameter_errors = {}
    Ne = len(score_vs_parameters)
    for i in range(Ne):
        for name in err_settings['variables'][i]:
            parameter_values = np.array(score_vs_parameters[i][name])
            score_values = np.array(score_vs_parameters[i]['score'])
            parameter_error = calculate_parameter_error(parameter_values, score_values, score_threshold)
            if name in parameter_errors:
                if not np.isnan(parameter_error) and not np.isnan(parameter_errors[name]):
                    if (parameter_error > parameter_errors[name]):
                        parameter_errors[name] = parameter_error
            else:
                parameter_errors[name] = parameter_error
    return [parameter_errors, numerical_error]

    
def calculate_numerical_error(best_parameters, err_settings, fit_settings, simulator, exp_data, spinA, spinB, calc_settings):
    # Create a generation
    Ns = err_settings['Ns']
    generation = Generation(Ns)
    generation.first_generation(fit_settings['parameters']['bounds'])
    # Set all genes to the optimized values
    for name in const['variable_names']:
        index = fit_settings['parameters']['indices'][name]
        if not (index == -1):
            for j in range(Ns):
                generation.chromosomes[j].genes[index] = best_parameters[name]['value']
    # Score the generation
    generation.score_chromosomes(fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
    # Sort chromosomes according to their score
    generation.sort_chromosomes()
    # Determine the variation of the score
    score_min = generation.chromosomes[0].score
    score_max = generation.chromosomes[Ns-1].score
    numerical_error = score_max - score_min
    return numerical_error


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
    if (parameter_error_lb+parameter_error_ub) < 0.95 * (parameter_min+parameter_max):
        if (lower_bound > parameter_min) or (upper_bound < parameter_max):
            if (parameter_error_lb > parameter_error_ub):
                parameter_error = parameter_error_lb
            else:
                parameter_error = parameter_error_ub  
    return parameter_error