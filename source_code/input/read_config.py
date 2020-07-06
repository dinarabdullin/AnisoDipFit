'''
Read the input data from the config file
'''

import io
import sys
import libconf
import numpy as np
from input.load_spectrum import load_spectrum
from input.load_timetrace import load_timetrace
from input.load_optimized_parameters import load_optimized_parameters
from input.symmetric_boundaries import symmetric_boundaries
from mathematics.find_max_on_interval import find_max_on_interval
from supplement.constants import const


def read_calculation_mode(config):
    mode = {}
    switch = int(config.mode)
    if (switch == 0):
        mode['simulation'] = 1
        mode['fitting'] = 0
        mode['error_analysis'] = 0
    elif (switch == 1):
        mode['simulation'] = 0
        mode['fitting'] = 1
        mode['error_analysis'] = 0
    elif (switch == 2):
        mode['simulation'] = 0
        mode['fitting'] = 0
        mode['error_analysis'] = 1
    else:
        raise ValueError('Illelgible value of mode!')
        sys.exit(1)
    return mode


def read_experimental_parameters(config):
    exp_data = {}  
    exp_data['path_spectrum'] = config.path_spectrum
    exp_data['path_timetrace'] = config.path_timetrace
    exp_data['t'] = []
    exp_data['sig'] = []
    exp_data['f'] = []
    exp_data['spc'] = []
    if not (exp_data['path_spectrum']==""):
        f, spc = load_spectrum(exp_data['path_spectrum'])
        exp_data['f'], exp_data['spc'] = symmetric_boundaries(f, spc)
    if not (exp_data['path_timetrace']==""):
        exp_data['t'], exp_data['sig'] = load_timetrace(exp_data['path_timetrace'])
    return exp_data


def read_spin_parameters(config):
    spinA = {}
    spinB = {}
    spinA['type'] = config.spinA.type
    spinA['g'] = np.array([float(config.spinA.g[0]), 
                           float(config.spinA.g[1]), 
                           float(config.spinA.g[2])])
    spinB['type'] = config.spinB.type
    spinB['g'] = np.array([float(config.spinB.g[0]), 
                           float(config.spinB.g[1]), 
                           float(config.spinB.g[2])])
    if (spinA['type'] == "anisotropic") and (spinB['type'] == "isotropic"):
        raise ValueError('Only SpinB can be anisotropic!')
        sys.exit(1)
    if (spinA['type'] == "anisotropic") and (spinB['type'] == "anisotropic"):
        raise ValueError('The case of two anisotropic spins is not supported yet!')
        sys.exit(1)
    return [spinA, spinB]


def read_simulation_settings(config, exp_data):
    sim_settings = {}
    sim_settings['modes'] = {}
    sim_settings['modes']['spc'] = int(config.simulation_modes.spc)
    sim_settings['modes']['timetrace'] = int(config.simulation_modes.timetrace)
    sim_settings['modes']['spc_vs_theta'] = int(config.simulation_modes.spc_vs_theta) 
    sim_settings['modes']['spc_vs_xi'] = int(config.simulation_modes.spc_vs_xi)
    sim_settings['modes']['spc_vs_phi'] = int(config.simulation_modes.spc_vs_phi)
    sim_settings['modes']['spc_vs_temp'] = int(config.simulation_modes.spc_vs_temp)
    sim_settings['parameters'] = {}
    sim_settings['parameters']['r_mean'] = float(config.simulation_parameters.r_mean)
    sim_settings['parameters']['r_width'] = float(config.simulation_parameters.r_width)
    sim_settings['parameters']['xi_mean'] = const['deg2rad'] * float(config.simulation_parameters.xi_mean)
    sim_settings['parameters']['xi_width'] = const['deg2rad'] * float(config.simulation_parameters.xi_width)
    sim_settings['parameters']['phi_mean'] = const['deg2rad'] * float(config.simulation_parameters.phi_mean)
    sim_settings['parameters']['phi_width'] = const['deg2rad'] * float(config.simulation_parameters.phi_width)
    sim_settings['parameters']['temp'] = float(config.simulation_parameters.temp)
    sim_settings['settings'] = {}
    sim_settings['settings']['theta_bins'] = []
    sim_settings['settings']['xi_bins'] = []
    sim_settings['settings']['phi_bins'] = []
    sim_settings['settings']['temp_bins'] = []
    if sim_settings['modes']['spc_vs_theta']:
        sim_settings['settings']['theta_bins'] = np.linspace(float(config.simulation_settings.theta_ranges[0]), 
                                                            float(config.simulation_settings.theta_ranges[1]), 
                                                            int(config.simulation_settings.theta_ranges[2]))
    if sim_settings['modes']['spc_vs_xi']:
        sim_settings['settings']['xi_bins'] = np.linspace(float(config.simulation_settings.xi_ranges[0]), 
                                                         float(config.simulation_settings.xi_ranges[1]), 
                                                         int(config.simulation_settings.xi_ranges[2]))
    if sim_settings['modes']['spc_vs_phi']:
        sim_settings['settings']['phi_bins'] = np.linspace(float(config.simulation_settings.phi_ranges[0]), 
                                                          float(config.simulation_settings.phi_ranges[1]), 
                                                          int(config.simulation_settings.phi_ranges[2]))
    if sim_settings['modes']['spc_vs_temp']:
        sim_settings['settings']['temp_bins'] = np.linspace(float(config.simulation_settings.temp_ranges[0]), 
                                                           float(config.simulation_settings.temp_ranges[1]), 
                                                           int(config.simulation_settings.temp_ranges[2]))
    if (exp_data['t'] == []):
        sim_settings['settings']['mod_depth'] = float(config.simulation_settings.mod_depth)
    else:
        sim_settings['settings']['mod_depth'] = 0
    if (exp_data['f'] == []):
        sim_settings['settings']['faxis_normalized'] = int(config.simulation_settings.faxis_normalized)
    else:
        sim_settings['settings']['faxis_normalized'] = 0
    sim_settings['settings']['plot_3d'] = int(config.simulation_settings.plot_3d)
    return sim_settings


def read_fitting_parameters(object):
    indices = {}
    bounds = []
    fixed = {}
    count = 0
    for name in const['variable_names']:
        attribute = getattr(object, name)
        vopt = int(attribute.opt)
        if vopt:
            indices[name] = count
            count += 1
            vmin = float(attribute.range[0]) * const['variable_scales'][name]
            vmax = float(attribute.range[1]) * const['variable_scales'][name]
            bounds.append([vmin, vmax])
        else:
            indices[name] = -1
            fixed[name] = float(attribute.value) * const['variable_scales'][name]
    return [indices, bounds, fixed, count]


def read_fitting_settings(config, exp_data): 
    fit_settings = {}
    fit_settings['parameters'] = {}
    fit_settings['parameters']['indices'] = {}
    fit_settings['parameters']['bounds'] = []
    fit_settings['parameters']['fixed'] = {}
    fit_settings['parameters']['indices'], fit_settings['parameters']['bounds'], fit_settings['parameters']['fixed'], fit_settings['parameters']['size'] = read_fitting_parameters(config.fitting_parameters)
    fit_settings['settings'] = {}
    fit_settings['settings']['fitted_data'] = config.fitting_settings.fitted_data
    fit_settings['settings']['display_graphics'] = int(config.fitting_settings.display_graphics)
    fit_settings['settings']['method'] = config.fitting_settings.method
    if (fit_settings['settings']['method'] == "genetic"):
        fit_settings['settings']['num_generations'] = int(config.fitting_settings.num_generations)
        fit_settings['settings']['size_generation'] = int(config.fitting_settings.size_generation)
        fit_settings['settings']['prob_crossover'] = float(config.fitting_settings.prob_crossover)
        fit_settings['settings']['prob_mutation'] = float(config.fitting_settings.prob_mutation)
    # Check the spin system
    if (fit_settings['settings']['fitted_data'] == 'spectrum') and (exp_data['f'] == []):
        raise ValueError('No spectrum was specified!')
        sys.exit(1)
    elif (fit_settings['settings']['fitted_data'] == 'timetrace') and (exp_data['t'] == []):
        raise ValueError('No time trace was specified!')
        sys.exit(1)
    return fit_settings


def read_error_analysis_settings(config, mode):
    err_settings = {}
    err_settings['variables'] = config.error_analysis.variables
    err_settings['Ns'] = int(config.error_analysis.Ns)
    err_settings['confidence_interval'] = float(config.error_analysis.confidence_interval)
    path_optimized_parameters = config.error_analysis.path_optimized_parameters
    if not (path_optimized_parameters == "") and (mode['error_analysis']):
        err_settings['optimized_parameters'] = load_optimized_parameters(path_optimized_parameters)
    return err_settings


def read_calculation_settings(config, exp_data):
    calc_settings = {}
    calc_settings['Ns'] = int(config.calculation_settings.Ns) 
    calc_settings['r_distr'] = config.calculation_settings.r_distr
    calc_settings['xi_distr'] = config.calculation_settings.xi_distr
    calc_settings['phi_distr'] = config.calculation_settings.phi_distr
    calc_settings['f_min'] = float(config.calculation_settings.fmin)
    calc_settings['f_max'] = float(config.calculation_settings.fmax)
    calc_settings['t_min'] = float(config.calculation_settings.tmin)
    calc_settings['t_max'] = float(config.calculation_settings.tmax)
    calc_settings['g_selectivity'] = int(config.calculation_settings.g_selectivity)
    if calc_settings['g_selectivity']:
        calc_settings['magnetic_field'] = float(config.calculation_settings.magnetic_field)
    calc_settings['noise_std'] = float(config.calculation_settings.noise_std)
    # Set the maximal amplitude of the spectrum and the maximal dipolar frequency
    if not (exp_data['f'] == []):
        if calc_settings['f_max']:
            calc_settings['spc_max'] = find_max_on_interval(exp_data['spc'], exp_data['f'], calc_settings['f_min'], calc_settings['f_max'])   
        else:
            calc_settings['f_max'] = np.amax(exp_data['f'])
            calc_settings['spc_max'] = find_max_on_interval(exp_data['spc'], exp_data['f'], calc_settings['f_min'], calc_settings['f_max'])       
    else:
        calc_settings['spc_max'] = 1.0
    # Set the maximal time point    
    if not (exp_data['t'] == []):
        if not calc_settings['t_max']:
            calc_settings['t_max'] = np.amax(exp_data['t'])
    return calc_settings


def read_output_settings(config):
    output_settings = {}
    output_settings['directory'] = config.output.directory
    output_settings['save_data'] = int(config.output.save_data)
    output_settings['save_figures'] = int(config.output.save_figures)
    return output_settings

    
def read_config(filepath):
    sys.stdout.write('\n') 
    sys.stdout.write('Reading out the configuration file... ') 
    mode = {}
    exp_data = {}
    spinA = {}
    spinB = {}
    sim_settings = {}
    fit_settings = {}
    err_settings = {}
    calc_settings = {}
    output_settings = {}
    with io.open(filepath) as file:
        config = libconf.load(file)
        mode = read_calculation_mode(config)
        exp_data = read_experimental_parameters(config)
        spinA, spinB = read_spin_parameters(config)
        if mode['simulation']:
            sim_settings = read_simulation_settings(config, exp_data)
        if mode['fitting'] or mode['error_analysis']:
            fit_settings = read_fitting_settings(config, exp_data)
            err_settings = read_error_analysis_settings(config, mode)
        calc_settings = read_calculation_settings(config, exp_data)
        output_settings = read_output_settings(config)
    sys.stdout.write('[DONE]\n\n')
    return [mode, exp_data, spinA, spinB, sim_settings, fit_settings, err_settings, calc_settings, output_settings]
    