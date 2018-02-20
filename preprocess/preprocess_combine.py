import pandas as pd
import os
from preprocess_iam import preprocess_iam_lines
from preprocess_bentham import preprocess_bentham

# run preprocessing scripts to get ensure data is in place
print("Running preprocessing on iam lines")
preprocess_iam_lines()
img_exists = os.path.exists("../data/BenthamDataset/img_size.txt")
alp_exists = os.path.exists("../data/BenthamDataset/alphabet.txt")
dir_exists = os.path.exists("../data/BenthamDataset/Images_mod")
if not (img_exists and alp_exists and dir_exists):
    print("Running preprocessing on Bentham")
    preprocess_bentham()

if not os.path.isdir("../data/combined/"):
    os.mkdir("../data/combined/")

# load in results of preprocessing
print()
print("Creating combined train.csv")
iam = pd.read_csv("../data/iamHandwriting/train.csv", sep="\t")
benth = pd.read_csv("../data/BenthamDataset/train.csv", sep="\t")
full = pd.concat([iam, benth])
full.to_csv("../data/combined/train.csv", sep="\t", index=False)

# get new max image size
print("Creating combined img_size")
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
print("Creating combined alphabet")
with open("../data/BenthamDataset/alphabet.txt", "r") as f:
    ab = f.readline()
with open("../data/iamHandwriting/alphabet.txt", "r") as f:
    ai = f.readline()
letters = {l for l in ab+ai}
with open("../data/combined/alphabet.txt", "w") as f:
    f.write("".join(sorted(letters)))