'''
Save the experimental and simulated PDS time traces
'''


def save_timetrace(t_sim, sig_sim, t_exp, sig_exp, filename):
    file = open(filename, 'w')
    if not (t_exp == []):
        for i in range(t_sim.size):
            file.write('{0:<12.4f} {1:<12.4f} {2:<12.4f} \n'.format(t_exp[i], sig_exp[i], sig_sim[i]))
    else:
        for i in range(t_sim.size):
            file.write('{0:<12.4f} {1:<12.4f} \n'.format(t_sim[i], sig_sim[i]))
    file.close()