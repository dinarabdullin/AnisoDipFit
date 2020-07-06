'''
Save the optimized fitting parameters
'''

from supplement.constants import const


def save_fitting_parameters(parameters, filename):    
    file = open(filename, 'w')
    file.write("{0:<16s} {1:<16s} {2:<16s} {3:<16s}\n".format('Parameter', 'Value', 'Optimized', 'Precision (+/-)'))
    for name in const['variable_names']:
        parameter = parameters[name]
        file.write("{0:<16s} ".format(parameter['longname']))
        file.write("{0:<16.3f} ".format(parameter['value']/const['variable_scales'][name]))
        file.write("{0:<16s} ".format(parameter['optimized']))
        file.write("{0:<16.3f}\n".format(parameter['precision']/const['variable_scales'][name]))
    file.close()