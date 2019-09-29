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

> 
is designed to work with the data set put in the Data folder as a text file. 
In this case, the data set contains information of the Top 200 Universities of the QS ranking. 

Having the data set, partitions where made in order to classify the Universities among two     
classes. Take into account if the code is going to be adapted to other data sets,   
it requires certain changes in the code.
