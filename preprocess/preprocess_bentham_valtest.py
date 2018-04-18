import numpy as np
import pandas as pd

import os
import warnings
import sys

from PIL import Image

# for printing during for loops
def my_print(text):
    sys.stdout.write(str(text) + "\t")
    sys.stdout.flush()

def preprocess_bentham():
    #### Read in data and modify data
    local_path = os.getcwd().replace("\\", "/")
    local_path = local_path.replace("preprocess", "")
    img_dir = "../data/BenthamDataset/Images/Lines/"
    trans_dir = "../data/BenthamDataset/Transcriptions/"
    part_dir = "../data/BenthamDataset/Partitions/"
    
    if not os.path.isdir("../data/BenthamTest/"):
        os.mkdir("../data/BenthamTest/")

    # get partitions (only use training for now)
    with open(os.path.join(part_dir, "TrainLines.lst")) as f:
        training = f.read().splitlines()
    with open(os.path.join(part_dir, "ValidationLines.lst")) as f:
        validation = f.read().splitlines()
    with open(os.path.join(part_dir, "TestLines.lst")) as f:
        test = f.read().splitlines()


    # filenames
    filenames = [f.replace(".txt", "") for f in os.listdir(trans_dir)]
    filenames = [f for f in filenames if f in validation or f in test]
    data_df = pd.DataFrame({"filenames": filenames})
    data_df["imgnames"] = [img_dir+f+".png" for f in data_df.filenames]
    data_df["transnames"] = [trans_dir+f+".txt" for f in data_df.filenames]


    # images
    print("Reading image files...")
    def readBenthamImg(fn):
        im = np.array(Image.open(fn))
        rep_val = max(np.median(im[:,:,0]), np.mean(im[:,:,0]))
        im[im[:,:,3] == 0] = [rep_val, rep_val, rep_val, 255]
        im = Image.fromarray(im).convert("L")
        return im
    img_list = []
    count = 0
    for f in data_df.imgnames:
        img_list.append(readBenthamImg(f))
        count += 1
        if count % 1000 == 0: my_print(count)
    data_df["images"] = img_list
    print()
    print(str(count) + " images read")


    data_df["imsizes"] = [np.array(i).shape for i in data_df.images]
    data_df["heights"] = [i[0] for i in data_df.imsizes]
    data_df["widths"] = [i[1] for i in data_df.imsizes]


    # transcriptions
    def readBenthamTrans(fn):
        return open(fn, "r").readline().replace("\n", "")
    data_df["transcription"] = [readBenthamTrans(f) for f in data_df.transnames]


    #### Get rid of unwanted rows
    # get rid of the really big images
    with open("../data/BenthamDataset/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        input_shape = (int(float(h)), int(float(w)))
    w95 = np.percentile(data_df.widths, 95)
    h95 = np.percentile(data_df.heights, 95)
    print("Max image size (width, height): ({0}, {1})".format(w95, h95))
    print("Max image size from training (width, height): ({0}, {1})".format(input_shape[1], input_shape[0]))
    data_df = data_df[np.logical_and(data_df.widths < input_shape[1], data_df.heights < input_shape[0])]
    with open("../data/BenthamTest/img_size.txt", "w") as f:
        f.write(",".join([str(w), str(h)]))

    # get rid of images with utf-8 special characters
    data_df["nonhex"] = [any([ord(i) > 127 for i in t]) 
                         for t in data_df.transcription]
    data_df = data_df[np.logical_not(data_df.nonhex)]


    #### Save new images
    new_img_dir = local_path + "/data/BenthamDataset/Images_mod/"
    if not os.path.isdir(new_img_dir):
        os.mkdir(new_img_dir)
    data_df["new_img_path"] = [new_img_dir + f+".png" for f in data_df.filenames]

    print("Saving modified images...")
    count = 0
    for i in data_df.index:
        data_df.loc[i, "images"].save(data_df.loc[i, "new_img_path"])
        count += 1
        if count % 1000 == 0: my_print(count)
    print()
    print(str(count) + " images saved")


    #### Save training file
    export_df = data_df[["new_img_path", "transcription"]]
    export_df.to_csv("../data/BenthamTest/train.csv", sep="\t", index=False)


    #### Find freqency of letters
    letters = dict()

    for tran in export_df.transcription:
        for l in list(tran):
            if l not in letters:
                letters[l] = 0
            letters[l] += 1
    letters = sorted(letters.items(), key = lambda f: f[1], reverse=True)
    with open("../data/BenthamDataset/alphabet.txt", "r") as f:
        alphabet = f.readline()
    with open("../data/BenthamTest/alphabet.txt", "w") as f:
        f.write(alphabet)

    print("\n")
    print("Letter freqencies:\n", letters)

if __name__ == "__main__":
    preprocess_bentham()