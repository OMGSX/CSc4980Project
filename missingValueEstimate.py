from pandas import read_csv
from sklearn.preprocessing import Imputer
import numpy as np

filepath = "./MissingData1.txt"

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

dataset = read_csv(filepath, sep='\t', header=None)

outFilePre = open('PettersMissingValue1Before.txt', 'w')

for item in dataset.values:
    outFilePre.write("%s \n" % item)

result = imputer.fit_transform(dataset.values)

print(result)

outFile = open('PettersMissingValue1.txt', 'w')

for item in result:
    outFile.write("%s \n" % item)
