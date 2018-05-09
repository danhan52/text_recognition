import numpy as np
import pandas as pd
import os
import sys
from preprocess_ASM_csv import *
from preprocess_bentham import *
from preprocess_iam import *
from preprocess_combined import *

# create the max image size file based on the image sizes in the data folders
# input is a list with the names of the folders to get the image sizes from
def get_img_sizes(folder_list = ["BenthamDataset", "BenthamTest",
                                "iamHandwriting", "iamTest"]):
    print("Creating modeling img_size")
    widths = []
    heights = []
    # read in image size files from all folders
    for fold in folder_list:
        with open("../data/{0}/img_size.txt".format(fold), "r") as f:
            w1, h1 = f.readline().split(",")
            widths.append(int(float(w1)))
            heights.append(int(float(h1)))

    # get the largest width and height separately
    w = np.max(widths)
    h = np.max(heights)
    with open("../data/img_size.txt", "w") as f:
        f.write(",".join([str(w), str(h)]))

# get new alphabet from the alphabets in the data folders
# input is a list with the names of the folders to get the alphabets from
def get_alphabet(folder_list = ["BenthamDataset", "BenthamTest",
                                "iamHandwriting", "iamTest"]):
    print("Creating modeling alphabet")
    alpha = ""
    # read alphabet strings
    for fold in folder_list:
        with open("../data/{0}/alphabet.txt".format(fold), "r") as f:
            alpha += f.readline()
        
    # combine alphabets, sort, and write
    letters = {l for l in alpha}
    with open("../data/alphabet.txt", "w") as f:
        f.write("".join(sorted(letters)))
        
# run all preprocessing scripts
def preprocess_all(preprocess_which="all", resize_to = 0.5):
    if preprocess_which == "all": # do bentham, iam, and ASM
        print("Creating all training data")
        preprocess_combined(True, resize_to)
        preprocess_combined(False, resize_to)
        get_img_sizes()
        get_alphabet()
        preprocess_ASM_csv()
    elif preprocess_which == "bentham": # do only benthma
        print("Creating bentham training data")
        preprocess_bentham(True, resize_to)
        preprocess_bentham(False, resize_to)
        get_img_sizes(["BenthamDataset", "BenthamTest"])
        get_alphabet(["BenthamDataset", "BenthamTest"])
    elif preprocess_which == "iam": # do only iam
        print("Creating iam training data")
        preprocess_iam_lines(True, resize_to)
        preprocess_iam_lines(False, resize_to)
        get_img_sizes(["iamHandwriting", "iamTest"])
        get_alphabet(["iamHandwriting", "iamTest"])
    elif preprocess_which == "combined": # do bentham and iam
        print("Creating combined training data")
        preprocess_combined(True, resize_to)
        preprocess_combined(False, resize_to)
        get_img_sizes()
        get_alphabet()
    elif preprocess_which == "asm": # do only asm
        print("Creating asm training data")
        preprocess_ASM_csv()
    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] not in ["all", "bentham", "iam", "combined", "asm"]:
            print("First argument must be one of 'all', 'bentham', 'iam', 'combined', or 'asm'")
        else:
            preprocess_all(sys.argv[1])
    elif len(sys.argv) == 3:
        if sys.argv[1] not in ["all", "bentham", "iam", "combined", "asm"]:
            print("First argument must be one of 'all', 'bentham', 'iam', 'combined', or 'asm'")
        else:
            preprocess_all(sys.argv[1], float(sys.argv[2]))
    else:
        preprocess_all()