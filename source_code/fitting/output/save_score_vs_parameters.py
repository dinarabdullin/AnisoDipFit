'''
Save the score in dependence of a single fitting parameter or a pair of fitting parameters
'''
	
from supplement.constants import const 

	
def save_score_vs_parameters(variables, score_vs_parameters, directory):
	Ne = len(variables)
	for i in range(Ne):
		# Number of genes in one set
		dim = len(variables[i])
		# Create a new file
		filename = directory + 'parameter_errors'
		for j in range(dim):
			filename += '-' + variables[i][j]
		filename += '.dat'
		file = open(filename, 'w')
		# Column names
		for j in range(dim):
			file.write("{0:<12s}".format(variables[i][j]))
		file.write("{0:<12s}".format('RMSD'))
		file.write("\n")
		# Column content
		Ns = len(score_vs_parameters[i]['score'])
		for k in range(Ns):
			for j in range(dim):
				name = variables[i][j]
				val = score_vs_parameters[i][name][k] / const['variable_scales'][name]
				file.write("{0:<12.6f}".format(val))
			file.write("{0:<12.6f}".format(score_vs_parameters[i]['score'][k]))
			file.write("\n")
		file.close()