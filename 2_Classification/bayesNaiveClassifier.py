from pandas import read_csv
from sklearn.preprocessing import Imputer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys
from io import StringIO

if len(sys.argv) != 5:
        print("[*] Bayes Naive Classifier\n"+
                "[*] Matt Petters, TeamMember2, TeamMember3, TeamMember4 \n" +
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

dataset = read_csv(filedata, sep='\t', header=None)
labelsData = read_csv(trainingLabelsFilename, sep='\t', header=None)
testData = read_csv(testDataFilename, sep='\t', header=None)

cleanData = imputer.fit_transform(dataset.values)

gnb = GaussianNB()

gnb.fit(cleanData, labelsData.values)

print(gnb.predict(testData.values))

