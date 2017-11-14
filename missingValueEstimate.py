from pandas import read_csv

import numpy

filepath = "./MissingData1.txt"

dataset = read_csv(filepath, sep='\t', header=None)

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(1.000000e+99, numpy.NaN)

dataset.fillna(dataset.mean(), inplace=True)

print(dataset.describe())
