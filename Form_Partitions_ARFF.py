import numpy as np
import math
import random
import os

fileInText = "Data_sets/Data.txt"
fileInArff = "Data_sets/Partition_Template_V2.arff"

for partition in range(50,151):
	#partition = 10
	count = 1
	fileOut = "Partitions_arff_V2/Partition_"+str(partition)+".arff"
	os.system("cp "+fileInArff+" "+fileOut)
	outfile = open(fileOut, "a+")
	classInstance = 0

	for line in open(fileInText, "r"):
		if(count > partition):
			classInstance = 1
		outfile.write(line[:line.find("\n")]+", "+str(classInstance)+"\n")
		count += 1

	outfile.close()