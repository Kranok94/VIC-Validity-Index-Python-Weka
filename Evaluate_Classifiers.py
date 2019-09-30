#---------------------------------------------------------------------------------------
# --------------------------------------                                                |
#| Author: Kevin Brian Kwan Chong Loo   |                                               |
#| ID: A01192177                        |                                               |
#| Date: September 30th, 2019           |                                               |
#| Email: kb.kwanloo@gmail.com          |                                               |
# --------------------------------------                                                |
#                                                                                       |
#The presented code evaluates data sets (imported as CSV files),                        |
#   and evaluates them based on different classifiers.                                  |
#                                                                                       |
#By default, the following classifiers are used and with their respective code:         |
#   - Bayesian Networks             --> BayesianNetwork                                 |
#   - Multi-Layer Perceptron        --> MultiLayerPerceptron                            |
#   - AdaBoost                      --> AdaBoost                                        |
#   - K Nearest Neighbor            --> KNN                                             |
#   - Random Forest                 --> RandomForest                                    |
#   - Support Vector Machines       --> SVM                                             |
#   - Naive Bayes                   --> NaiveBayes                                      |
#   - Linear Discriminant Analysis  --> LDA                                             |
#                                                                                       |
#If only certain classifiers from the list above want to be used,                       |
#   it only has to be an argument when executing this Python Code.                      |
#   To specify the classifiers that want to be used, the codes of                       |
#   each classifier must be put as an argument separated with a comma                   |
#   since all the classifiers to be used are read as a string and                       |
#   parsed by the comma.                                                                |
#                                                                                       |
#As for the Cross Validation value, by default, it has a value of 10.                   |
#   Likewise, the values of K-Cross Validation can be specified as an                   |
#   argument.                                                                           |
#                                                                                       |
# --------------------                                                                  |
#|Example of Execution|                                                                 |
# --------------------                                                                  |
# $ python3 Evaluate_Classifiers.py -c SVM,NaiveBayes,LDA -k 5                          |
#                                                                                       |
#For the previous example, only Support Vector Machines, Naive Bayes and                |
#   Linear Discriminant Analysis are going to be used to evaluate a data set,           |
#   with 5-Cross Validation.                                                            |
#                                                                                       |
#For the classifiers, the "sklearn" Python library can be used. However, an alternative |
#   is set is a GPU wants to be used in order to run the code faster. "h2o4gpu" has     |
#   almost the same functions as "sklearn".                                             |
#                                                                                       |
#Likewise, a Python-Weka library was installed in order to use certain classifiers      |
#   from Weka and receive the information in Python.                                    |
#                                                                                       |
# ----------------------                                                                |
#|Description of Problem|                                                               |
# ----------------------                                                                |
#This code is designed to work with the data set put in the data folder. The data set   |
#   contains information of the Top 200 Universities of the QS ranking. Having the      |
#   data set, partitions where made in order to classify the Universities among two     |
#   classes. Take into account if the code is going to be adapted to other data sets,   |
#   it requires certain changes in the code.                                            |
#---------------------------------------------------------------------------------------


#Basic Libraries to be used
#--------------------------
import numpy as np
from scipy import interp
import pandas as pd
import time
from statistics import mean
import argparse
import os
import subprocess

#Import Libraries of Classifiers using sklearn (uncomment if it wants to be used)
#---------------------------------------------
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import roc_curve, auc, roc_auc_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier


#Import Libraries of Classifiers using an alternative version of sklearn that used GPU
#-------------------------------------------------------------------------------------
#   Install h2o4gpu: https://github.com/h2oai/h2o4gpu
#   Note: requires Python 3.6 to run
#   $ python3.6 Evaluate_Classifiers_V8.py -c SVM,NaiveBayes,LDA -k 5
from h2o4gpu.svm import SVC
from h2o4gpu.naive_bayes import GaussianNB
from h2o4gpu.discriminant_analysis import LinearDiscriminantAnalysis
from h2o4gpu.metrics import roc_curve, auc, roc_auc_score
from h2o4gpu.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from h2o4gpu.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from h2o4gpu.neural_network import MLPClassifier

#Import Libraries to use Weka with Python
#----------------------------------------
from weka.classifiers import Classifier, Evaluation
import weka.core.converters as converters
from weka.core.classes import Random
import weka.core.jvm as jvm
from weka.filters import Filter

#In order to evaluate the data sets, it is required to have the file in CSV and ARFF format.
folderPathOfArffFiles = "Partitions_arff/"
folderPathOfCSVFiles = "Partitions_csv/"

#Arguments to receive as input in the execution of this Python Code.
arguments = argparse.ArgumentParser()
arguments.add_argument("-c", "--classifiers", default="BayesianNetwork,MultiLayerPerceptron,AdaBoost,KNN,RandomForest,SVM,NaiveBayes,LDA",
    help="List of Classifiers separated by ',' without spaces.")
arguments.add_argument("-k", "--kfold", default=10,
    help="Value of K-Fold.")
args = vars(arguments.parse_args())

kFold = int(args["kfold"])

#--------------------------------------------------------------------------------------------------
#Method to obtain the ROC-AUC using Bayesian Networks in Weka. This function only receives the name
#   of the file to be analyzed without the extension format.
def obtainBayesNet(file):
    #The path of the arff extension file must be put.
    data = converters.load_any_file(folderPathOfArffFiles+file+".arff")

    #In the case of this specific data set, the first two attributes were removed since they
    #   represent the name and ranking which are unique values that would affect the classification.
    #   Depending on the data set, certain attributes must be removed.
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R","1-2"])
    remove.inputformat(data)
    data = remove.filter(data)
    #It is specified that the class value is the last attribute.
    data.class_is_last()

    #Define the classifier to be used.
    classifier = Classifier(classname="weka.classifiers.bayes.BayesNet")
    evaluation = Evaluation(data)                     
    evaluation.crossvalidate_model(classifier, data, kFold, Random(42))

    #The ROC-AUC is extracted from the string that is received from Weka.
    info = evaluation.class_details()
    roc_area = float(info[406:411])

    return roc_area

#---------------------------------------------------------------------------------------------------
#This function is similar to the previous one, with the slight change of the classifier to be used.
#In the case of wanting to use more classifiers in Weka, only the "classname" must be changed.
def obtainSVM(file):
    data = converters.load_any_file(folderPathOfArffFiles+file+".arff")
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R","1-2"])
    remove.inputformat(data)
    data = remove.filter(data)
    data.class_is_last()

    classifier = Classifier(classname="weka.classifiers.functions.LibSVM")
    evaluation = Evaluation(data)                     
    evaluation.crossvalidate_model(classifier, data, kFold, Random(42))

    info = evaluation.class_details()
    roc_area = float(info[406:411])

    return roc_area


#-------------------------------------------------------------------------------------------
#This line is required to use Weka with Python.
jvm.start(packages=True)

#Additional Information for Knowing the time it takes to classify all the Partitions. 
startProgram = time.time()

#Assignment of K-Fold Cross Validation.
cv = StratifiedKFold(n_splits=kFold)

#Extract the classifiers that want to be used.
classifiersList = args["classifiers"].split(",")

#Name of the output file where the performance of each classifier and the maximum among them
#   per partition is saved in a file.
fileOut = "Cluster_Evaluations_"+str(kFold)+"_Fold.txt"
outfile = open(fileOut, "a+")
outfile.write("Partition_File,"+str(args["classifiers"])+",Maximum\n")
outfile.close()     

#Obtain the list of the partitions that are going to be tested with each classifier.
command = "ls "+folderPathOfCSVFiles
r = subprocess.check_output(command, shell=True)
rS = str(r,'utf-8')
files = rS.split()

#Loop that goes through each file.
for input_file in files:
    information = input_file+","
    outfile = open(fileOut, "a+")
    filePath = folderPathOfCSVFiles+input_file

    #Reading the partition file.
    data = pd.read_csv(filePath, header = 0)
    data = data._get_numeric_data()
    attribute_Names = list(data.columns.values)

    #Define the attributes and class in separate variables.
    X = np.array(data[attribute_Names[1:len(attribute_Names)-1]])
    y = np.array(data["class_94"])

    classifiersMean = []
    print("Analyzing %s" %(filePath))
    startExecution = time.time()
    for classifierType in classifiersList:
        meanAUCArray = []
        #K iterations of the K-Fold Cross Validation
        for kIterations in range(0,kFold):
            print("Classifier: %s Iteration: %d" %(classifierType,kIterations))
            valid = True

            #Selection of the classifier that wants to be used.
            if(classifierType == "SVM"):
                #For Support Vector Machine, the classifier of sklearn or Weka can be
                #   used, just by commenting the following lines.

                #classifier = SVC(kernel='linear', probability=True)        #sklearn
                meanAUC = obtainSVM(input_file[0:input_file.find(".")])     #Weka
                meanAUCArray.append(meanAUC)                                #Weka
                valid = False                                               #Weka
            elif(classifierType == "NaiveBayes"):
                classifier = GaussianNB()
            elif(classifierType == "LDA"):
                classifier = LinearDiscriminantAnalysis()
            elif(classifierType == "RandomForest"):
                classifier = RandomForestClassifier()
            elif(classifierType == "KNN"):
            	classifier = KNeighborsClassifier(5)
            elif(classifierType == "AdaBoost"):
            	classifier = AdaBoostClassifier()
            elif(classifierType == "MultiLayerPerceptron"):
            	classifier = MLPClassifier()
            elif(classifierType == "BayesianNetwork"):
                meanAUC = obtainBayesNet(input_file[0:input_file.find(".")])
                meanAUCArray.append(meanAUC)
                valid = False
            else:
                print("%s is not available. This classifier will be ignored. "%(classifierType))
                valid = False

            #Obtaining the Cross Validation ROC-AUC per classifier. If Weka is used for the
            #   classifier, the following code is omitted. 
            if(valid):
                i = 0
                aucs = []
                print("Starting K-Fold Cross Validation...")
                #Split per Cross Validation is applied.
                for train, test in cv.split(X, y):
                    start = time.time()
                    print("Training CV-%d..." %(i))
                    #The classifier is trained with the data and tested with the test set.
                    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                    #ROC-AUC is obtained.
                    auc = roc_auc_score(y[test], probas_[:, 1])
                    print("Cross Validation: %d AUC: %3.2f Time: %5.3fs" %(i,auc,time.time()-start))
                    #Each AUC per Cross Validation of each classifier is saved.
                    aucs.append(auc)
                    i+=1

                #The AUC per Cross Validation of each classifier are averaged.
                meanAUC = mean(aucs)
                meanAUCArray.append(meanAUC)
                print("File: %s Classifier: %s Mean AUC: %5.3f\n" %(input_file,classifierType,meanAUC))

        kIterationMeanAUC = mean(meanAUCArray)
        classifiersMean.append(kIterationMeanAUC)
        information+=str(round(kIterationMeanAUC,5))+","

    #The maximum AUC value is obtained among all classifiers and writen in a text file.
    print("Maximum AUC among Classifiers: %5.3f" %(max(classifiersMean)))
    print("Total Time of Execution: %6.3fs" %(time.time()-startExecution))
    information+=str(round(max(classifiersMean),5))+"\n"
    outfile.write(information)
    outfile.close()

jvm.stop()
print("\nComplte Program Time of Execution: %6.3fs" %(time.time()-startProgram))
