# VIC-Validity-Index-Python-Weka
Implementation of set of classifiers in Python and Weka

#---------------------------------------------------------------------------------------
# --------------------------------------                                                |
#| Author: Kevin Brian Kwan Chong Loo   |                                               |
#| ID: A01192177                        |                                               |
#| Data: September 30th, 2019           |                                               |
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
