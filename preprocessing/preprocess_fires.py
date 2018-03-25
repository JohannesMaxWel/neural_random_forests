# -*- coding: utf-8 -*-
import numpy as np

def load_data():
    filename = "./datasets/data/forest_fires/forest_fires.forestfires.csv"
    with open(filename, 'r') as f:
        rawtext = f.read()

    lines = rawtext.split('\n')
    lines = lines[1:-1]# discard first line (headers) and last (empty)
    splitted = [ line.split(",")[:2]+line.split(",")[4:] for line in lines]   #discard column 3 and 4 (categorical)
    N = len(splitted)
    M = len(splitted[0])
    X = np.zeros([N,M-1])
    Y = np.zeros([N,1])

    for i, row in enumerate(splitted):
        #convert into floats
        num = [float(string) for string in row]

        # each column is one feature, the last column is the target to predict.
        X[i,:] = num[:-1]
        Y[i] = num[-1]

    return X,Y





if __name__ == "__main__":
    X,Y = load_data()
