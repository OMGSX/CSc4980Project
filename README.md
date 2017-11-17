# Machine Learning Project
 

Requires Python 3

To install dependencies using `pip`:

`pip3 install pandas sklearn numpy`


## 1. Missing Value Estimator

To run directly:

`python3 missingValueEstimate.py path/to/input.txt path/to/output.txt`

or run:

`sh estimateSampleOne.sh`

`sh estimateSampleTwo.sh`

`sh estimateSampleThree.sh`

To run the provided sample data.


## 2. Classification

To run directly (training dataset 1 shown here): 

`python3 classifiers/bayesNaiveClassifier.py src/TrainData1.txt src/TrainLabel1.txt src/TestData1.txt out/TestDataLabels1.txt`

or run

`sh trainAndTest1.sh`

`sh trainAndTest2.sh`

`sh trainAndTest3.sh`

`sh trainAndTest4.sh`

`sh trainAndTest5.sh`

### NOTE: Due to widely varying input, we chose to implement 5 different classifiers which expect different specific formats for training data and test data. Keep this in mind when trying to run your own arbitrary input. The shell scripts (ie; trainAndTest1.sh) show examples of which classifiers to use with which kinds of data
