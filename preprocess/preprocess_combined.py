import numpy as np
import pandas as pd
import os
import sys
from preprocess_bentham import *
from preprocess_iam import *



# preprocess the combined iam and bentham training set
def preprocess_combined(is_training=True, resize_to=0.5, print_letters=False):
    # run preprocessing scripts to get ensure data is in place
    print("\nRunning preprocessing on iam lines training")
    preprocess_iam_lines(resize_to=resize_to, is_training=is_training, print_letters=print_letters)
    print("\nRunning preprocessing on Bentham training")
    preprocess_bentham(resize_to=resize_to, is_training=is_training, print_letters=print_letters)

    if is_training:
        if not os.path.isdir("../data/combined_train/"):
            os.mkdir("../data/combined_train/")

        # load in results of preprocessing
        print("\nCreating combined train.csv")
        iam = pd.read_csv("../data/iamHandwriting/train.csv", sep="\t")
        benth = pd.read_csv("../data/BenthamDataset/train.csv", sep="\t")
        full = pd.concat([iam, benth])
        full.to_csv("../data/combined_train/train.csv", sep="\t", index=False)
    else:
        if not os.path.isdir("../data/combined_test/"):
            os.mkdir("../data/combined_test/")

        # load in results of preprocessing
        print("\nCreating combined train.csv")
        iam = pd.read_csv("../data/iamTest/train.csv", sep="\t")
        benth = pd.read_csv("../data/BenthamTest/train.csv", sep="\t")
        full = pd.concat([iam, benth])
        full.to_csv("../data/combined_test/train.csv", sep="\t", index=False)


# preprocess all the combined data
if __name__ == "__main__":
    if len(sys.argv >= 3):
        resize_to = float(sys.argv[2])
    else:
        resize_to = 0.5

    if len(sys.argv >= 4):
        print_letters = sys.argv[3] == "True"
    else:
        print_letters = False

    if len(sys.argv) >= 2:
        if sys.argv[1] == "train":
            preprocess_combined(True, resize_to, print_letters)
        elif sys.argv[1] == "test":
            preprocess_combined(False, resize_to, print_letters)
        elif sys.argv[1] == "both":
            preprocess_combined(True, resize_to, print_letters)
            preprocess_combined(False, resize_to, print_letters)
        else:
            print("First argument must be one of 'train', 'test', or 'both'")
    else:
        preprocess_combined(True)
        preprocess_combined(False)
