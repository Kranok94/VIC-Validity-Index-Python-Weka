#---------------------------------------------------------------------------------------
# --------------------------------------                                                |
#| Author: Kevin Brian Kwan Chong Loo   |                                               |
#| ID: A01192177                        |                                               |
#| Date: September 30th, 2019           |                                               |
#| Email: kb.kwanloo@gmail.com          |                                               |
# --------------------------------------                                                |
#                                                                                       |
#The presented code generates partitions files in .csv format in order to be used in 	|
#	with sklearn in Python. It requires a data set as an input in text file format.  	|
#	Additionally, a template in .csv format must be provided with the name of each  	|
#	attribute.  																		|
#---------------------------------------------------------------------------------------

import numpy as np
import math
import random
import os

#Name of file with the original data set.
fileInText = "Data/Dataset.txt"
#Name of file of the template in .csv format with the name of the attributes. 
fileInTemplate = "Data/Partition_Template.csv"

#Define the lower and upper limits of the partitions.
lowerLimitPartition = 50
upperLimitPartition = 150

for partition in range(lowerLimitPartition,upperLimitPartition+1):
	count = 1
	#Output file path and name of partition file.
	fileOut = "Partitions_csv/Partition_"+str(partition)+".csv"
	os.system("cp "+fileInTemplate+" "+fileOut)
	outfile = open(fileOut, "a+")

	#Based on the value of the partition, the instances obtain their respective class value.
	classInstance = 0

	for line in open(fileInText, "r"):
		if(count > partition):
			classInstance = 1
		outfile.write(line[:line.find("\n")]+", "+str(classInstance)+"\n")
		count += 1

	outfile.close()
