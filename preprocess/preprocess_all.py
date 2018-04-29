import numpy as np
import pandas as pd
import os
import sys
from preprocess_ASM_csv import *
from preprocess_bentham import *
from preprocess_iam import *


# preprocess the combined iam and bentham training set
def preprocess_combine_train(resize_to=1.0):
    # run preprocessing scripts to get ensure data is in place
    print("\nRunning preprocessing on iam lines training")
    preprocess_iam_lines(resize_to=resize_to, is_training=True)
    print("\nRunning preprocessing on Bentham training")
    preprocess_bentham(resize_to=resize_to, is_training=True)

    if not os.path.isdir("../data/combined_train/"):
        os.mkdir("../data/combined_train/")

    # load in results of preprocessing
    print("\nCreating combined train.csv")
    iam = pd.read_csv("../data/iamHandwriting/train.csv", sep="\t")
    benth = pd.read_csv("../data/BenthamDataset/train.csv", sep="\t")
    full = pd.concat([iam, benth])
    full.to_csv("../data/combined_train/train.csv", sep="\t", index=False)

# preprocess the combined iam and bentham test set
def preprocess_combine_test(resize_to=1.0):
    # run preprocessing scripts to get ensure data is in place
    print("\nRunning preprocessing on iam lines test")
    preprocess_iam_lines(resize_to=resize_to, is_training=False)
    print("\nRunning preprocessing on Bentham test")
    preprocess_bentham(resize_to=resize_to, is_training=False)

    if not os.path.isdir("../data/combined_test/"):
        os.mkdir("../data/combined_test/")

    # load in results of preprocessing
    print("\nCreating combined train.csv")
    iam = pd.read_csv("../data/iamTest/train.csv", sep="\t")
    benth = pd.read_csv("../data/BenthamTest/train.csv", sep="\t")
    full = pd.concat([iam, benth])
    full.to_csv("../data/combined_test/train.csv", sep="\t", index=False)

# preprocess all the combined data
def preprocess_combine(resize_to=1.0):
    preprocess_combine_train(resize_to)
    preprocess_combine_test(resize_to)


# get new max image size
def get_img_sizes():
    print("Creating combined img_size")
    with open("../data/BenthamDataset/img_size.txt", "r") as f:
        w1, h1 = f.readline().split(",")
        w1 = int(float(w1))
        h1 = int(float(h1))
    with open("../data/BenthamTest/img_size.txt", "r") as f:
        w2, h2 = f.readline().split(",")
        w2 = int(float(w2))
        h2 = int(float(h2))
    with open("../data/iamHandwriting/img_size.txt", "r") as f:
        w3, h3 = f.readline().split(",")
        w3 = int(float(w3))
        h3 = int(float(h3))
    with open("../data/iamTest/img_size.txt", "r") as f:
        w4, h4 = f.readline().split(",")
        w4 = int(float(w4))
        h4 = int(float(h4))
        
    w = np.max([w1, w2, w3, w4])
    h = np.max([h1, h2, h3, h4])
    with open("../data/img_size.txt", "w") as f:
        f.write(",".join([str(w), str(h)]))

# get new alphabet
def get_alphabet(do_ASM=True):
    print("Creating combined alphabet")
    with open("../data/BenthamDataset/alphabet.txt", "r") as f:
        a1 = f.readline()
    with open("../data/BenthamTest/alphabet.txt", "r") as f:
        a2 = f.readline()
    with open("../data/iamHandwriting/alphabet.txt", "r") as f:
        a3 = f.readline()
    with open("../data/iamTest/alphabet.txt", "r") as f:
        a4 = f.readline()
    if do_ASM:
        with open("../data/ASM/alphabet.txt", "r") as f:
            a5 = f.readline()
    else:
        a5 = ""
        
    letters = {l for l in a1+a2+a3+a4+a5}
    with open("../data/alphabet.txt", "w") as f:
        f.write("".join(sorted(letters)))
        
# run all preprocessing scripts
def preprocess_all(resize_to = 1.0, do_ASM=True):
    preprocess_combine(resize_to)
    if do_ASM:
        preprocess_ASM_csv()
    
    get_img_sizes()
    get_alphabet(do_ASM)
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        preprocess_all(float(sys.argv[1]))
    elif len(sys.argv) >= 3:
        preprocess_all(float(sys.argv[1]), sys.argv == "True")
    else:
        preprocess_all()