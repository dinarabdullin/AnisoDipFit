'''
Estimation of noise
'''

from mathematics.rmsd import rmsd
from fitting.generation import Generation
from supplement.constants import const
from fitting.score_function import get_fit


def calculate_noise_std(fit, exp_data, fit_settings, calc_settings):
    noise_std = 0
    if calc_settings['noise_std'] == 0:
        # If the standard deviation of noise is set to 0, calculate it
        if fit_settings['settings']['fitted_data'] == 'spectrum':
            noise_std = rmsd(fit, exp_data['spc'], exp_data['f'], calc_settings['f_min'], calc_settings['f_max'])
        elif fit_settings['settings']['fitted_data'] == 'timetrace':
            noise_std = rmsd(fit, exp_data['sig'], exp_data['t'], calc_settings['t_min'], calc_settings['t_max'])
    else:
        noise_std = calc_settings['noise_std']
    return noise_std


def calculate_fit_and_noise_std(best_parameters, fit_settings, simulator, exp_data, spinA, spinB, calc_settings):
    # Create a generation with one chromosome and set all genes to the optimized values   
    generation = Generation(1)
    generation.first_generation(fit_settings['parameters']['bounds'])   
    for name in const['variable_names']:
        index = fit_settings['parameters']['indices'][name]
        if not (index == -1):
            generation.chromosomes[0].genes[index] = best_parameters[name]['value']
    # Calculate the corresponding fit
    fit = get_fit(generation.chromosomes[0].genes, fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
    # Calculate the standard deviation of noise
    noise_std = calculate_noise_std(fit, exp_data, fit_settings, calc_settings)
    return noise_std