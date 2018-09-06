from sklearn.datasets import load_files       
#from keras.utils import np_utils
import numpy as np
from glob import glob
import csv
from shutil import copyfile

code_dict = {}
pure_code_dict = {}
pure_code_cnt = 0


# Load all the images
with open('img_code.csv', 'rb') as csvfile:
    img_code_reader = csv.reader(csvfile, delimiter=',')
    for row in img_code_reader:
        if len(row) == 2:
            pure_code_cnt = pure_code_cnt + 1
            code = row[1]
            if code in pure_code_dict.keys():
                pure_code_dict[code] = pure_code_dict[code] + 1
            else:
                pure_code_dict[code] = 1
            print "{} {}".format(row[0], row[1])
            try:
                if row[1] == '14':
                    copyfile("./all-images/" + row[0] + ".ppm", "./14_or_not/14/" + row[0] + ".ppm")
                else:
                    copyfile("./all-images/" + row[0] + ".ppm", "./14_or_not/not_14/" + row[0] + ".ppm")
            except Exception as e:
                print "Exception caught"
        for ind, code in enumerate(row):
            if ind > 0:
                if code in code_dict.keys():
                    code_dict[code] = code_dict[code] + 1
                else:
                    code_dict[code] = 1

