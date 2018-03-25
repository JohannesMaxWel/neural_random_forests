# -*- coding: utf-8 -*-
import numpy as np

def load_data():
    filename = "datasets/data/wisconsin/wpbc.data.txt"
    with open(filename, 'r') as f:
        rawtext = f.read()

    lines = rawtext.split('\n')
    lines = lines[:-1]# discard last line (empty)
    splitted = [ line.split(",")[2:] for line in lines]   #from column 3 onwards
    #N = len(splitted)
    N = 194
    M = len(splitted[0])
    X = np.zeros([N,M-1])
    Y = np.zeros([N,1])

    i = 0
    for row in splitted:
        #convert into floats
        try:
            num = [float(string) for string in row]
            # each column is one feature, the last column is the target to predict.
            X[i,:] = num[1:]
            Y[i] = num[0]
            i +=1
        except ValueError:
            print ("Discarding row {}: non-numeric input.".format(i))

    return X,Y
