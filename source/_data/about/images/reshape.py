import cv2
import numpy as np
from sys import argv


if(len(argv) < 2):
	print("please set arguments")
else:
	_,name = argv
	input = cv2.imread(name)
	output = cv2.resize(input, (400, 300))
	prefix = name.split(".")[0]
	suffix = name.split(".")[-1]
	outname = prefix + "_400_300." + suffix
	cv2.imwrite(outname, output)