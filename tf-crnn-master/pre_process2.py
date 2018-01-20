# get the details to make all images the same size
import os
import warnings
import numpy as np

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
    

with open("data/Partitions/TestLines.lst") as f:
    test = f.read().splitlines()

local_path = "/home/danny/Repos/text_recognition/tf-crnn-master/"
count = 0
    for fn in filenames:
        new_fn = local_path + img_dir_new + fn + ".png"
        trans = readBenthamTrans(fn)
        trans = trans.replace('"', '""')
        if any([ord(i) > 127 for i in trans]):
            os.remove(new_fn)
        elif fn in test:
            os.remove(new_fn)