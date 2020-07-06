'''
Save the results of simulation
'''

import sys
from simulation.output.save_spectrum import save_spectrum
from simulation.output.save_timetrace import save_timetrace
from simulation.output.save_spectrum_vs_theta import save_spectrum_vs_theta
from simulation.output.save_spectrum_vs_parameter import save_spectrum_vs_parameter
from supplement.constants import const


def save_simulation_data(simulator, sim_settings, exp_data, output_settings):
    if output_settings['save_data']:
        sys.stdout.write('Saving the simulation results into the directory:\n')
        sys.stdout.write(output_settings['directory'])
        if sim_settings['modes']['spc']:
            filename = output_settings['directory'] + 'spc.dat'
            save_spectrum(simulator.f, simulator.spc, exp_data['f'], exp_data['spc'], filename)
        if sim_settings['modes']['timetrace']:
            filename = output_settings['directory'] + 'timetrace.dat'
            save_timetrace(simulator.t, simulator.sig, exp_data['t'], exp_data['sig'], filename)
            filename = output_settings['directory'] + 'timetrace_sim.dat'
            save_timetrace(simulator.t, simulator.sig, [], [], filename)
        if sim_settings['modes']['spc_vs_theta']:
            filename = output_settings['directory'] + 'spc_vs_theta.dat'
            save_spectrum_vs_theta(simulator.f, simulator.theta_bins, simulator.spc_vs_theta, filename)
        if sim_settings['modes']['spc_vs_xi']:
            filename = output_settings['directory'] + 'spc_vs_xi.dat'
            save_spectrum_vs_parameter(simulator.f, simulator.xi_bins, simulator.spc_vs_xi, filename)
        if sim_settings['modes']['spc_vs_phi']:
            filename = output_settings['directory'] + 'spc_vs_phi.dat'
            save_spectrum_vs_parameter(simulator.f, simulator.phi_bins, simulator.spc_vs_phi, filename) 
        if (sim_settings['modes']['spc_vs_temp'] and not simulator.spc_vs_temp == []):
            filename = output_settings['directory'] + 'spc_vs_temp.dat'
            save_spectrum_vs_parameter(simulator.f, simulator.temp_bins, simulator.spc_vs_temp, filename)
            filename = output_settings['directory'] + 'depth_vs_temp.dat'
            save_spectrum_vs_parameter(simulator.g, simulator.temp_bins, simulator.depth_vs_temp, filename)
        sys.stdout.write(' [DONE]\n\n')