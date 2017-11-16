from pandas import read_csv
from sklearn.preprocessing import Imputer
import numpy as np
import sys
from io import StringIO

if len(sys.argv) != 3:
        print("[*] Missing Value Estimator\n"+
                "[*] Matt Petters, Dylan Monroe, Kristian Sandoval, Prithvi Ravi \n" +
            "[*] Usage: python3 missingValueEstimate.py input.txt output.txt\n"+
                "[*] Error: Invalid number of arguments")
        sys.exit(0)
inputFilename = sys.argv[1]
outputFilename = sys.argv[2]

with open(inputFilename, 'r') as file :
  filedata = file.read()

filedata = filedata.replace('1.00000000000000e+99', 'NaN')

filedata = StringIO(filedata)

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

dataset = read_csv(filedata, sep='\t', header=None)

result = imputer.fit_transform(dataset.values)

print(result)

outFile = open(outputFilename, 'w')

for item in result:
    outFile.write("%s \n" % item)
