# -*- coding: utf-8 -*-
import numpy as np
import xlrd

def load_data():
    filename = "./datasets/data/concrete/Concrete_Data.xls"
    book = xlrd.open_workbook(filename)

    # get the first worksheet
    first_sheet = book.sheet_by_index(0)
    first_sheet.nrows
    first_sheet.ncols

    rows = []
    for irow in range (first_sheet.nrows):
        rows += [first_sheet.row_values(irow)]

    N = len(rows) - 1 # n_samples
    M = len(rows[0]) -1 #n_features
    X = np.zeros([N,M])
    Y = np.zeros([N,1])
    for irow, row in enumerate(rows[1:]):
        X[irow,:] = row[:-1]
        Y[irow] = row[-1]
    return X,Y
