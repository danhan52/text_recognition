
# coding: utf-8

# In[1]:

# get the details to make all images the same size
import os
import warnings
import numpy as np


local_path = os.getcwd()
img_dir = "/data/Images_mod/"
trans_dir = "./data/Transcriptions/"
part_dir = "./data/Partitions/"

filenames = [f.replace(".txt", "") for f in os.listdir(trans_dir)]


# # Create transcription csvs
# Deal with utf-8 compliance issues

# read the transcription
def readBenthamTrans(fn, ftype=".txt", trans_dir=trans_dir):
    with open(trans_dir + fn + ftype, "r") as f:
        trans = f.readline()
        trans = trans.replace("\n", "")
    return trans


with open(os.path.join(part_dir, "TrainLines.lst")) as f:
    training = f.read().splitlines()

with open(os.path.join(part_dir, "ValidationLines.lst")) as f:
    validation = f.read().splitlines()

with open(os.path.join(part_dir, "TestLines.lst")) as f:
    test = f.read().splitlines()


count = 0
alphabet = dict()
with open("data/train.csv", "w") as f_tr, open("data/valid.csv", "w") as f_va, open("data/test.csv", "w") as f_te:
    for fn in filenames:
        new_fn = local_path + img_dir + fn + ".png"
        trans = readBenthamTrans(fn)
        
        # get the alphabet too
        for l in list(trans):
            if l not in alphabet:
                alphabet[l] = 0
            alphabet[l] += 1

        # write the csvs
        trans = trans.replace('"', '""')
        if any([ord(i) > 127 for i in trans]):
            continue
        if fn in training:
            f_tr.write("{0}\t\"{1}\"\n".format(new_fn, trans))
        elif fn in validation:
            f_va.write("{0}\t\"{1}\"\n".format(new_fn, trans))
        elif fn in test:
            f_te.write("{0}\t\"{1}\"\n".format(new_fn, trans))

        if count > 1000:
            break
        count += 1
        if count % 1000 == 0:
            print(count, end="\t")

print()
alpha_string1 = "".join(np.sort(list(alphabet)))
print(alpha_string1)

alphabet2 = {k for k, v in alphabet.items() if ord(k) <= 127}
alpha_string2 = "".join(np.sort(list(alphabet2)))
print(alpha_string2)
