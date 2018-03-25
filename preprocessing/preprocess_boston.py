# -*- coding: utf-8 -*-
import numpy as np

def load_data():
    filename = "./datasets/data/boston_housing/housing.data.txt"
    with open(filename, 'r') as f:
        rawtext = f.read()

    lines = rawtext.split('\n')
    lines = [ line[1:] for line in lines]   #strip first empty symbol
    N = len(lines) - 1
    M = len(lines[0].split("  "))
    X = np.zeros([N,M])
    Y = np.zeros([N,1])

    for i, line in enumerate(lines[:-1]):
        strings = [x for x in line.split(' ') if x != ""]
        num = [float(string) for string in strings]

        Y[i] = num[-1]
        X[i,:] = num[:-1]

    return X,Y
