# get the details to make all images the same size
import os
import warnings
import numpy as np
from skimage import io as skimio
from skimage import color as skimcolor
import skimage.transform as skimtrans
import matplotlib.pyplot as plt

local_path = "/home/danny/Repos/text_recognition/tf-crnn-master/"
img_dir_old = "data/Images/"
img_dir_new = "data/Images_mod/"
trans_dir = "data/Transcriptions/"

filenames = [f.replace(".txt", "") for f in os.listdir(trans_dir)]

# read the transcription
def readBenthamTrans(fn, ftype=".txt", trans_dir=trans_dir):
    with open(trans_dir + fn + ftype, "r") as f:
        trans = f.readline()
        trans = trans.replace("\n", "")
    return trans
    

with open("data/Partitions/TrainLines.lst") as f:
    training = f.read().splitlines()

with open("data/Partitions/ValidationLines.lst") as f:
    validation = f.read().splitlines()

with open("data/Partitions/TestLines.lst") as f:
    test = f.read().splitlines()

local_path = "/home/danny/Repos/text_recognition/tf-crnn-master/"
count = 0
with open("data/train.csv", "w") as f_tr, open("data/valid.csv", "w") as f_va, open("data/test.csv", "w") as f_te:
    for fn in filenames:
        new_fn = local_path + img_dir_new + fn + ".png"
        trans = readBenthamTrans(fn)
        trans = trans.replace('"', '""')
        if any([ord(i) > 127 for i in trans]):
            continue
#         trans = "".join([i if ord(i) < 128 else chr(126) for i in trans])
        if fn in training:
            f_tr.write("{0}\t\"{1}\"\n".format(new_fn, trans))
        elif fn in validation:
            f_va.write("{0}\t\"{1}\"\n".format(new_fn, trans))
        elif fn in test:
            f_te.write("{0}\t\"{1}\"\n".format(new_fn, trans))