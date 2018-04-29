import numpy as np
from preprocess_ASM_csv import *
from preprocess_bentham import *
from preprocess_iam import *
from preprocess_combine import *

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
        
    w = np.max([w1, w2, w3])
    h = np.max([h1, h2, h3])
    with open("../data/img_size.txt", "w") as f:
        f.write(",".join([str(w), str(h)]))

# get new alphabet
def get_alphabet():
    print("Creating combined alphabet")
    with open("../data/BenthamDataset/alphabet.txt", "r") as f:
        a1 = f.readline()
    with open("../data/BenthamTest/alphabet.txt", "r") as f:
        a2 = f.readline()
    with open("../data/iamHandwriting/alphabet.txt", "r") as f:
        a3 = f.readline()
    with open("../data/ASM/alphabet.txt", "r") as f:
        a4 = f.readline()
        
    letters = {l for l in a1+a2+a3+a4}
    with open("../data/alphabet.txt", "w") as f:
        f.write("".join(sorted(letters)))
        
# run all preprocessing scripts
def preprocess_all(resize_to = 1.0):
    preprocess_combine(resize_to)
    preprocess_bentham(resize_to, False)
    preprocess_ASM_csv()
    
    get_img_sizes()
    get_alphabet()
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        preprocess_all(float(sys.argv[1]))
    else:
        preprocess_all()