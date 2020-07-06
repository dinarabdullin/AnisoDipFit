'''
Score function
'''

import numpy as np
from mathematics.chi2 import chi2
from spinphysics.flip_probabilities import flip_probabilities
from supplement.constants import const


def set_parameters(parameters, indices, fixed):
	all_parameters = {}
	for name in const['variable_names']:
		index = indices[name]
		if not (index == -1):
			all_parameters[name] = parameters[index]
		else:
			all_parameters[name] = fixed[name]
	return all_parameters

	
def score_function(parameters, fit_settings, simulator, exp_data, spinA, spinB, calc_settings):
    # Set parameters
    indices = fit_settings['parameters']['indices']
    fixed = fit_settings['parameters']['fixed']
    all_parameters = set_parameters(parameters, indices, fixed)
    # Calculate the temperature-based weights
    weights_temp = []
    if (not (indices['temp'] == -1) and calc_settings['g_selectivity'] and (spinB['type'] == "anisotropic")):
        weights_temp = flip_probabilities(simulator.gB, spinB['g'], calc_settings['magnetic_field'], all_parameters['temp'])
    # Calculate dipolar frequencies
    fdd, theta, weights_xi = simulator.dipolar_frequencies(all_parameters, spinA, spinB) 
    # Calulate the total weights
    weights = simulator.total_weights(weights_temp, weights_xi)
    if fit_settings['settings']['fitted_data'] == 'spectrum':
        # Calculate the dipolar spectrum
        fit = simulator.dipolar_spectrum(fdd, weights)
        # Calculate the score
        score = chi2(fit, exp_data['spc'], exp_data['f'], calc_settings['f_min'], calc_settings['f_max'], calc_settings['noise_std'])         
    elif fit_settings['settings']['fitted_data'] == 'timetrace':
        # Calculate the dipolar timetrace
        #fit = simulator.dipolar_timetrace(fdd, weights)
        fit = simulator.dipolar_timetrace_fast(fdd, weights)
        # Calculate the score
        score = chi2(fit, exp_data['sig'], exp_data['t'], calc_settings['t_min'], calc_settings['t_max'], calc_settings['noise_std'])
    return score

	
def get_fit(parameters, fit_settings, simulator, exp_data, spinA, spinB, calc_settings):
    # Set parameters
    indices = fit_settings['parameters']['indices']
    fixed = fit_settings['parameters']['fixed']
    all_parameters = set_parameters(parameters, indices, fixed)
    # Calculate the temperature-based weights
    weights_temp = []
    if (not (indices['temp'] == -1) and calc_settings['g_selectivity'] and (spinB['type'] == "anisotropic")):
        weights_temp = flip_probabilities(simulator.gB, spinB['g'], calc_settings['magnetic_field'], all_parameters['temp'])
    # Calculate dipolar frequencies
    fdd, theta, weights_xi = simulator.dipolar_frequencies(all_parameters, spinA, spinB) 
    # Calulate the total weights
    weights = simulator.total_weights(weights_temp, weights_xi)
    # Calculate the dipolar spectrum or the dipolar time trace
    if fit_settings['settings']['fitted_data'] == 'spectrum':
        # Calculate the dipolar spectrum
        fit = simulator.dipolar_spectrum(fdd, weights)        
    elif fit_settings['settings']['fitted_data'] == 'timetrace':
        # Calculate the dipolar timetrace
        #fit = simulator.dipolar_timetrace(fdd, weights)
        fit = simulator.dipolar_timetrace_fast(fdd, weights)
    return fit