import numpy as np
import math
import random
import os

fileInText = "Data_sets/Data.txt"
fileInTemplate = "Data_sets/Partition_Template.csv"

for partition in range(50,151):
	count = 1
	fileOut = "Partitions_csv_V1/Partition_"+str(partition)+".csv"
	os.system("cp "+fileInTemplate+" "+fileOut)
	outfile = open(fileOut, "a+")
	classInstance = 0

	for line in open(fileInText, "r"):
		if(count > partition):
			classInstance = 1
		outfile.write(line[:line.find("\n")]+", "+str(classInstance)+"\n")
		count += 1

	outfile.close()
