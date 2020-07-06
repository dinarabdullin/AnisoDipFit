'''
Save the fit to the experimental data
'''

from simulation.output.save_spectrum import save_spectrum
from simulation.output.save_timetrace import save_timetrace

	
def save_fit(fit, data, fitted_data, filename=''):
    if fitted_data == 'spectrum':
        save_spectrum(data['f'], fit, data['f'], data['spc'], filename)
    elif fitted_data == 'timetrace':
        save_timetrace(data['t'], fit, data['t'], data['sig'], filename)