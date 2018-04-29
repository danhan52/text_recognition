import numpy as np
from preprocess_iam import *

# get new max image size
def get_img_sizes():
    print("Creating combined img_size")
    with open("../data/iamHandwriting/img_size.txt", "r") as f:
        w1, h1 = f.readline().split(",")
        w1 = int(float(w1))
        h1 = int(float(h1))
    with open("../data/iamTest/img_size.txt", "r") as f:
        w2, h2 = f.readline().split(",")
        w2 = int(float(w2))
        h2 = int(float(h2))
        
    w = np.max([w1, w2])
    h = np.max([h1, h2])
    with open("../data/img_size.txt", "w") as f:
        f.write(",".join([str(w), str(h)]))

# get new alphabet
def get_alphabet():
    print("Creating combined alphabet")
    with open("../data/iamHandwriting/alphabet.txt", "r") as f:
        a1 = f.readline()
    with open("../data/iamTest/alphabet.txt", "r") as f:
        a2 = f.readline()
        
    letters = {l for l in a1+a2}
    with open("../data/alphabet.txt", "w") as f:
        f.write("".join(sorted(letters)))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        preprocess_iam_lines(float(sys.argv[1]), True)
        preprocess_iam_lines(float(sys.argv[1]), False)
    else:
        preprocess_iam_lines(1.0, True)
        preprocess_iam_lines(1.0, False)
    get_img_sizes()
    get_alphabet()

