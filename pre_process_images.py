
# coding: utf-8

# In[1]:

# get the details to make all images the same size
import os
import warnings
import numpy as np
from skimage import io as skimio
from skimage import color as skimcolor
import skimage.transform as skimtrans
import matplotlib.pyplot as plt
import pickle

img_dir_old = "data/Images/"
img_dir_new = "data/Images_mod/"
img_dir_pkl = "data/Images_np/"
trans_dir = "data/Transcriptions/"

filenames = [f.replace(".txt", "") for f in os.listdir(trans_dir)]


# # Create transcription csvs
# Deal with utf-8 compliance issues

# In[2]:

# read the transcription
def readBenthamTrans(fn, ftype=".txt", trans_dir=trans_dir):
    with open(trans_dir + fn + ftype, "r") as f:
        trans = f.readline()
        trans = trans.replace("\n", "")
    return trans


# In[3]:

with open("data/Partitions/TrainLines.lst") as f:
    training = f.read().splitlines()

with open("data/Partitions/ValidationLines.lst") as f:
    validation = f.read().splitlines()

with open("data/Partitions/TestLines.lst") as f:
    test = f.read().splitlines()


# In[4]:

#local_path = "/home/danny/Repos/text_recognition/tf-crnn-master/"
count = 0
with open("data/train.csv", "w") as f_tr, open("data/valid.csv", "w") as f_va, open("data/test.csv", "w") as f_te:
    for fn in filenames:
        new_fn = img_dir_new + fn + ".png"
        trans = readBenthamTrans(fn)
        trans = trans.replace('"', '""')
        if any([ord(i) > 127 for i in trans]):
            continue
        if fn in training:
            f_tr.write("{0}\t\"{1}\"\n".format(new_fn, trans))
        elif fn in validation:
            f_va.write("{0}\t\"{1}\"\n".format(new_fn, trans))
        elif fn in test:
            f_te.write("{0}\t\"{1}\"\n".format(new_fn, trans))


# In[ ]:

# get the alphabet
count = 0
alphabet = dict()
for fn in filenames:
    trans = readBenthamTrans(fn)
    for l in list(trans):
        if l not in alphabet:
            alphabet[l] = 0
        alphabet[l] += 1
    if count % 1000 == 0: print(count, end="\t")
    count += 1

print()
alpha_string1 = "".join(np.sort(list(alphabet)))
print(alpha_string1)

alphabet2 = {k for k, v in alphabet.items() if v > 1}
alpha_string2 = "".join(np.sort(list(alphabet2)))
print(alpha_string2)


# In[ ]:

for l in alpha_string1:
    print(l, ord(l), end="\t")


# # Pad images
# To do: pad in tensorflow

# In[ ]:

sizes = []
count = 0
for fn in filenames:
    im = skimio.imread(img_dir_old + fn + ".png")
    sizes.append(im.shape)
    if count % 1000 == 0: print(count, end = "\t")
    count += 1


# In[ ]:

heights = [s[0] for s in sizes]
widths = [s[1] for s in sizes]
max_height = np.max(heights)
max_width = np.max(widths)
print(max_height, max_width)
med_height = np.median(heights)
med_width = np.median(widths)
print(med_height, med_width)


# In[ ]:

# read the image file
def readBenthamImg(fn, ftype=".png", img_dir_new=img_dir_new,
                   img_dir_old=img_dir_old, goal_size = (med_height, med_width)):
    im = skimio.imread(img_dir_old + fn + ftype)
    # turn all transparent pixels to background (average)
    rep_val = max(np.median(im[:,:,0]), np.mean(im[:,:,0]))
    im[im[:,:,3] == 0] = [rep_val, rep_val, rep_val, 255]
    im = skimcolor.rgb2gray(im)*255

    # make all images the same size
    size = im.shape
    if int(1.0*goal_size[1]/size[1]*size[0]) < goal_size[0]:
        new_size = (round(1.0*goal_size[1]/size[1]*size[0]), goal_size[1])
        pad_size = ((0, int(goal_size[0] - new_size[0])), (0,0))
    elif int(1.0*goal_size[0]/size[0]*size[1]) < goal_size[1]:
        new_size = (goal_size[0], round(1.0*goal_size[0]/size[0]*size[1]))
        pad_size = ((0,0), (0, int(goal_size[1] - new_size[1])))
    elif all(np.equal(size, goal_size)):
        new_size = goal_size
        pad_size = ((0,0), (0,0))
    # resize and pad based on previous calculation
    im = skimtrans.resize(im, new_size, mode="constant")
    im = np.pad(im, pad_size, mode="constant", constant_values=rep_val)
    im = im.astype("int")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimio.imsave(img_dir_new + fn + ftype, im)
    return img_dir_new + fn + ftype


# In[ ]:

with open("data/Partitions/TrainLines.lst") as f:
    training = f.read().splitlines()

with open("data/Partitions/ValidationLines.lst") as f:
    validation = f.read().splitlines()

with open("data/Partitions/TestLines.lst") as f:
    test = f.read().splitlines()


# In[ ]:

count = 0
for fn in filenames:
    new_fn = readBenthamImg(fn)
    if count % 1000 == 0: print(count, end = "\t")
    count += 1

