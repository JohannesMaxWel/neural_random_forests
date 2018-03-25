# -*- coding: utf-8 -*-
import numpy as np

def load_data():
    filename = "./datasets/data/crimes/communities.data.txt"
    with open(filename, 'r') as f:
        lines = [x.rstrip() for x in f.readlines()]

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    numbers = [float(x) if isfloat(x) else -1111.1 for x in lines[0].split(',')]

    N = len(lines)
    M = len(numbers) - 1
    X = np.zeros([N,M])
    Y = np.zeros([N,1])


    for i, line in enumerate(lines):
        numbers = [float(x) if isfloat(x) else -1111.1 for x in line.split(',')]
        Y[i] = numbers[-1]
        X[i,:] = numbers[:-1]

    # remove these columns as they contain missing data points
    cc = list(set(np.where(X==-1111.1)[1]))
    rem = list(set(range(M)).difference(cc))
    X_new = X[:,rem]

    return X_new,Y
