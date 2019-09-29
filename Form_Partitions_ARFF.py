#---------------------------------------------------------------------------------------
# --------------------------------------                                                |
#| Author: Kevin Brian Kwan Chong Loo   |                                               |
#| ID: A01192177                        |                                               |
#| Data: September 30th, 2019           |                                               |
#| Email: kb.kwanloo@gmail.com          |                                               |
# --------------------------------------                                                |
#                                                                                       |
#The presented code generates partitions files in .arff format in order to be used in 	|
#	Weka. It requires a data set as an input in text file format. Additionally, a  		|
#	template must be provided in .arff format with the type of attribute and possible  	|
#	values for nominal attributes.														|
#---------------------------------------------------------------------------------------

import numpy as np
import math
import random
import os

#Input file of the original data set.
fileInText = "Data/Dataset.txt"
#Input file of template in .arff format with the attributes' name and possible values.
fileInArff = "Data/Partition_Template.arff"

#Define the lower and upper limits of the partitions.
lowerLimitPartition = 50
upperLimitPartition = 150

for partition in range(lowerLimitPartition,upperLimitPartition+1):
	count = 1
	#Output file path and name of each partition.
	fileOut = "Partitions_arff/Partition_"+str(partition)+".arff"
	os.system("cp "+fileInArff+" "+fileOut)
	outfile = open(fileOut, "a+")

	#Writing the partition with its corresponding class value depending on the instance rank.
	classInstance = 0

	for line in open(fileInText, "r"):
		if(count > partition):
			classInstance = 1
		outfile.write(line[:line.find("\n")]+", "+str(classInstance)+"\n")
		count += 1

	outfile.close()