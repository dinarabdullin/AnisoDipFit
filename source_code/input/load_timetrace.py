'''
Load an experimental time trace
'''

import numpy as np


def load_timetrace(filepath):
	x_list = []
	y_list = []
	file = open(filepath, 'r')
	for line in file:
		str = line.split()
		x_list.append(float(str[0]))
		y_list.append(float(str[1]))
	x_array = np.array(x_list)	
	y_array = np.array(y_list)	
	return [x_array, y_array]