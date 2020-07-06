'''
Check whether a function y(x) has the boundaries symmetric to x=0
'''


def symmetric_boundaries(x, y):
    xsym = []
    ysim = []
    if (x[0] == -x[-1]):
        xsym = x
        ysym = y
    if (x[0] < -x[0]):
        xsym = x[1:]
        ysym = y[1:]
    if (x[0] > -x[0]):
        xsym = x[:-1]
        ysym = y[:-1]    
    return [xsym, ysym]