'''
Plot of the results of simulation
'''

import sys
import numpy as np
from simulation.graphics.plot_spectrum import plot_spectrum
from simulation.graphics.plot_spectrum_vs_theta import plot_spectrum_vs_theta
from simulation.graphics.plot_spectrum_vs_parameter_3d import plot_spectrum_vs_parameter_3d
from simulation.graphics.plot_spectrum_vs_parameter_2d import plot_spectrum_vs_parameter_2d
from simulation.graphics.plot_timetrace import plot_timetrace
from simulation.graphics.plot_pb_vs_temp import plot_pb_vs_temp
from supplement.constants import const


def plot_simulation_data(simulator, sim_settings, exp_data, calc_settings, output_settings):
    sys.stdout.write('Plotting the simulation results... ')
    if sim_settings['modes']['spc']:
        filename = output_settings['directory'] + 'spc.png'
        plot_spectrum(simulator.f, simulator.spc, exp_data['f'], exp_data['spc'], 
                      sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings, 
                      output_settings['save_figures'], filename)
    if sim_settings['modes']['timetrace']:
        filename = output_settings['directory'] + 'timetrace.png'
        plot_timetrace(simulator.t, simulator.sig, exp_data['t'], exp_data['sig'], 
                       output_settings['save_figures'], filename)  
    if sim_settings['modes']['spc_vs_theta']:
        filename = output_settings['directory'] + 'spc_vs_theta.png'
        plot_spectrum_vs_theta(simulator.f, simulator.spc, simulator.theta_bins, simulator.spc_vs_theta, 
                               sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings, 
                               output_settings['save_figures'], filename)
    if sim_settings['modes']['spc_vs_xi']:
        filename = output_settings['directory'] + 'spc_vs_xi.png'
        parameter_label = r'$\mathit{\xi}$ ' + u'(\N{DEGREE SIGN})'
        if sim_settings['settings']['plot_3d']:
            plot_spectrum_vs_parameter_3d(simulator.f, simulator.xi_bins, simulator.spc_vs_xi, 
                                          sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings,
                                          output_settings['save_figures'], filename, parameter_label, True)
        else:
            plot_spectrum_vs_parameter_2d(simulator.f, simulator.xi_bins, simulator.spc_vs_xi,  
                                          sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings,
                                          output_settings['save_figures'], filename, parameter_label, False)
    if sim_settings['modes']['spc_vs_phi']:
        filename = output_settings['directory'] + 'spc_vs_phi.png'
        parameter_label = r'$\mathit{\phi}$ '  + u'(\N{DEGREE SIGN})'
        if sim_settings['settings']['plot_3d']:
            plot_spectrum_vs_parameter_3d(simulator.f, simulator.phi_bins, simulator.spc_vs_phi, 
                                          sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings, 
                                          output_settings['save_figures'], filename, parameter_label, False)
        else:
            plot_spectrum_vs_parameter_2d(simulator.f, simulator.phi_bins, simulator.spc_vs_phi, 
                                          sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings, 
                                          output_settings['save_figures'], filename, parameter_label, False)   
    if (sim_settings['modes']['spc_vs_temp'] and not simulator.spc_vs_temp == []):
        filename = output_settings['directory'] + 'spc_vs_temp.png'
        parameter_label = r'$\mathit{Temperature (K)}$'
        if sim_settings['settings']['plot_3d']:
            plot_spectrum_vs_parameter_3d(simulator.f, simulator.temp_bins, simulator.spc_vs_temp,  
                                          sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings,
                                          output_settings['save_figures'], filename, parameter_label, False)
        else:
            plot_spectrum_vs_parameter_2d(simulator.f, simulator.temp_bins, simulator.spc_vs_temp,  
                                          sim_settings['settings']['faxis_normalized'], simulator.fn, calc_settings,
                                          output_settings['save_figures'], filename, parameter_label, False)  
    sys.stdout.write('[DONE]\n\n')