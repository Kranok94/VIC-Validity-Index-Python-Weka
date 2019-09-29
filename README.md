# VIC-Validity-Index-Python-Weka
Implementation of set of classifiers in Python and Weka

Author: **Kevin Brian Kwan Loo**              
Date of creation: September 30th, 2019                                                          
Email: **kb.kwanloo@gmail.com**  


**Purpose of Project**                                                               
> The present project validates data sets using VIC. VIC is a validity index where, given a set
of classifiers, it applies K-Fold Cross Validation with ROC-Area Under the Curve (AUC). After evaluating all the
classifiers, the resulting validity index is the highest among all the classifiers.

**Implementation**
> The focus of this project is evaluating a data set holding information of the Top 200 Universities of the
QS World Ranking in 2019. In this case partitions are made in order to evaluate variations of the data set.
The partition Python scripts take the input data set and assigns a binary classification value based on the
ranking of the University. For example, a partition of 75 would assing all the Universities from Rank 1 to
Rank 75 the value of 0 and the Universities with Rank 76 to 200 with the value of 1. The partitions are made
in both .arff and .csv format.

> Having the partitions, the script that evaluates the classifiers is applied where it takes the files located in
the partitions folders of both CSV and ARFF. The following classifiers can be used:
  - Bayesian Networks             --> BayesianNetwork                                 
  - Multi-Layer Perceptron        --> MultiLayerPerceptron                            
  - AdaBoost                      --> AdaBoost                                        
  - K Nearest Neighbor            --> KNN                                             
  - Random Forest                 --> RandomForest                                    
  - Support Vector Machines       --> SVM                                             
  - Naive Bayes                   --> NaiveBayes                                      
  - Linear Discriminant Analysis  --> LDA         
  
> By default, all the classifiers are used with 10-Fold Cross Validation. If only certain classifiers want to be used
their respective name as presented on the right of the arrow must be put as an argument when running the script. If many classifiers want to be used, they have to be put together separated by a comma. Likewise, the value of the K-Fold can be changed from the default value. An example execution is shown as followed:
```
$ python3 Evaluate_Classifiers.py -c SVM,NaiveBayes,LDA -k 5 
```
                                            
