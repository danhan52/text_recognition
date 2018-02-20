import pandas as pd
import os
from preprocess_iam import preprocess_iam_lines
from preprocess_bentham import preprocess_bentham

# run preprocessing scripts to get ensure data is in place
preprocess_iam_lines()
preprocess_bentham()

if not os.path.isdir("../data/combined/"):
    os.mkdir("../data/combined/")

# load in results of preprocessing
iam = pd.read_csv("../data/iamHandwriting/train.csv", sep="\t")
benth = pd.read_csv("../data/BenthamDataset/train.csv", sep="\t")
full = pd.concat([iam, benth])
full.to_csv("../data/combined/train.csv", sep="\t", index=False)

# get new max image size
with open("../data/BenthamDataset/img_size.txt", "r") as f:
    wb, hb = f.readline().split(",")
    wb = int(float(wb))
    hb = int(float(hb))
with open("../data/iamHandwriting/img_size.txt", "r") as f:
    wi, hi = f.readline().split(",")
    wi = int(float(wi))
    hi = int(float(hi))

w = max(wb, wi)
h = max(hb, hi)
with open("../data/combined/img_size.txt", "w") as f:
    f.write(",".join([str(w), str(h)]))

# get new alphabet
with open("../data/BenthamDataset/alphabet.txt", "r") as f:
    ab = f.readline()
with open("../data/iamHandwriting/alphabet.txt", "r") as f:
    ai = f.readline()

letters = {l for l in ab+ai}
with open("../data/combined/alphabet.txt", "w") as f:
    f.write("".join(letters))