import pandas as pd
import os
import sys
from preprocess_iam import preprocess_iam_lines
from preprocess_bentham import preprocess_bentham


def preprocess_combine(resize_to=1.0):
    # run preprocessing scripts to get ensure data is in place
    print("Running preprocessing on iam lines")
    preprocess_iam_lines(resize_to=resize_to)
    print("Running preprocessing on Bentham")
    preprocess_bentham(resize_to=resize_to)

    if not os.path.isdir("../data/combined/"):
        os.mkdir("../data/combined/")

    # load in results of preprocessing
    print()
    print("Creating combined train.csv")
    iam = pd.read_csv("../data/iamHandwriting/train.csv", sep="\t")
    benth = pd.read_csv("../data/BenthamDataset/train.csv", sep="\t")
    full = pd.concat([iam, benth])
    full.to_csv("../data/combined/train.csv", sep="\t", index=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preprocess_combine(float(sys.argv[1]))
    else:
        preprocess_combine()