from pandas import read_csv
from sklearn.preprocessing import Imputer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys
from io import StringIO

if len(sys.argv) != 5:
        print("[*] Bayes Naive Classifier\n"+
              "[*] Matt Petters, Dylan Monroe, Kristian Sandoval, Prithvi Ravi \n" +
              "[*] Usage: python3 bayesNaiveClassifier.py trainingData.txt trainingLabels.txt testData.txt outputLabels.txt\n"+
              "[*] Error: Invalid number of arguments")
        sys.exit(0)
trainingDataFilename = sys.argv[1]
trainingLabelsFilename = sys.argv[2]
testDataFilename = sys.argv[3]
outputLabelsFilename = sys.argv[4]

with open(trainingDataFilename, 'r') as file :
  filedata = file.read()

filedata = filedata.replace('1.00000000000000e+99', 'NaN')

filedata = StringIO(filedata)

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

dataset = read_csv(filedata, delim_whitespace=True, header=None)
# print(dataset.values)
labelsData = np.loadtxt(trainingLabelsFilename)
# print(labelsData)

testData = read_csv(testDataFilename,delim_whitespace=True,header=None)
# print(testData.values)

cleanData = imputer.fit_transform(dataset.values)

gnb = GaussianNB()

gnb.fit(cleanData, labelsData)

result = gnb.predict(testData.values)

outFile = open(outputLabelsFilename, 'w')

for item in result:
    outFile.write("%s \n" % item)
