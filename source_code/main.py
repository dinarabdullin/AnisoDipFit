'''
Main file of AnisoDipFit
'''

import argparse
from input.read_config import read_config
from simulation.simulation import Simulator
from simulation.graphics.plot_simulation_data import plot_simulation_data
from simulation.output.save_simulation_data import save_simulation_data
from fitting.genetic_algorithm import GeneticAlgorithm
from fitting.noise_estimation import calculate_fit_and_noise_std
from fitting.graphics.plot_fitting_data import plot_fitting_data
from fitting.graphics.plot_error_analysis_data import plot_error_analysis_data
from fitting.output.save_fitting_data import save_fitting_data
from fitting.output.save_error_analysis_data import save_error_analysis_data
from supplement.make_output_directory import make_output_directory 
from supplement.keep_figures_live import keep_figures_live 


if __name__ == '__main__':
    # Read out the config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="A path to a configuration file")
    args = parser.parse_args()
    config_path = args.filepath
    mode, exp_data, spinA, spinB, sim_settings, fit_settings, err_settings, calc_settings, output_settings = read_config(config_path)

    # Make an output directory
    make_output_directory(output_settings, config_path)

    # Spectral simulator
    simulator = Simulator(calc_settings)

    # Simulation
    if mode['simulation']:
        # Run the simulation
        simulator.run_simulation(sim_settings, exp_data, spinA, spinB, calc_settings)	
        # Save simulation results
        save_simulation_data(simulator, sim_settings, exp_data, output_settings)	
        # Plot simulation results 
        plot_simulation_data(simulator, sim_settings, exp_data, calc_settings, output_settings)

    # Fitting
    if mode['fitting']:
        # Init the fitting mode of the simulator
        simulator.init_fitting(fit_settings, exp_data, spinA, spinB, calc_settings)
        if fit_settings['settings']['method'] == "genetic":
            # Optimizer
            optimizer = GeneticAlgorithm(fit_settings['settings'], exp_data)
            # Run the fitting
            optimizer.run_optimization(fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
        # Save the fitting results
        save_fitting_data(optimizer, exp_data, fit_settings, output_settings)
        # Plot the fitting results
        plot_fitting_data(optimizer, exp_data, fit_settings, calc_settings, output_settings)
        # Run the error analysis
        optimizer.error_analysis(err_settings, fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
        # Save the results of the error analysis 
        save_error_analysis_data(optimizer, err_settings, output_settings)
        # Plot the results of the error analysis 
        plot_error_analysis_data(optimizer, err_settings, fit_settings, output_settings)
        # Display the optmized fitting parameters
        optimizer.print_optimized_parameters()    

    # error analysis
    if mode['error_analysis']:
        # Init the fitting mode of the simulator
        simulator.init_fitting(fit_settings, exp_data, spinA, spinB, calc_settings)
        # Optimizer
        if fit_settings['settings']['method'] == "genetic":
            optimizer = GeneticAlgorithm(fit_settings['settings'], exp_data)
        # Set the best parameters
        optimizer.best_parameters = err_settings['optimized_parameters']
        # Check that the variance of noise is not 0
        if calc_settings['noise_std'] == 0:
            calc_settings['noise_std'] = calculate_fit_and_noise_std(optimizer.best_parameters, fit_settings, simulator, exp_data, spinA, spinB, calc_settings)       
        # Run the error analysis
        optimizer.error_analysis(err_settings, fit_settings, simulator, exp_data, spinA, spinB, calc_settings)
        # Save the results of the error analysis
        save_error_analysis_data(optimizer, err_settings, output_settings)
        # Plot the results of the error analysis 
        plot_error_analysis_data(optimizer, err_settings, fit_settings, output_settings)
        # Display the optmized fitting parameters
        optimizer.print_optimized_parameters()
           
    # Keep all figures live
    keep_figures_live()