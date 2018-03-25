# -*- coding: utf-8 -*-
import numpy as np

def load_data():
    filename = "./datasets/data/protein/CASP.csv"
    with open(filename, 'r') as f:
        lines = [x.strip() for x in f.readlines()]

    data = [eval(line) for line in lines[1:]]    # first row is header info
    N = len(data)
    M = len(data[0][1:])
    assert N == 45730
    assert M == 9

    X = np.zeros([N,M])
    Y = np.zeros([N,1])

    for i, row in enumerate(data):
        Y[i] = row[0]
        X[i,:] = row[1:]

    return X,Y


if __name__ == "__main__":
    X,Y = load_data()
    print(X.shape)
    print(Y.shape)
