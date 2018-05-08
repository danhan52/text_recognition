import numpy as np
import pandas as pd

import os
import warnings
import sys

from PIL import Image


def preprocess_bentham(is_training=True, resize_to=0.5, print_letters=False):
    #### Read in data and modify data
    local_path = os.getcwd().replace("\\", "/")
    local_path = local_path.replace("preprocess", "")
    img_dir = "../data/BenthamDataset/Images/Lines/"
    trans_dir = "../data/BenthamDataset/Transcriptions/"
    part_dir = "../data/BenthamDataset/Partitions/"

    # get directory to send all info to
    write_dir = "../data/"
    if is_training:
        write_dir = write_dir + "BenthamDataset/"
    else:
        write_dir = write_dir + "BenthamTest/"
        
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    # get partitions
    with open(os.path.join(part_dir, "TrainLines.lst")) as f:
        training = f.read().splitlines()
    with open(os.path.join(part_dir, "ValidationLines.lst")) as f:
        validation = f.read().splitlines()
    with open(os.path.join(part_dir, "TestLines.lst")) as f:
        test = f.read().splitlines()


    # filenames
    filenames = [f.replace(".txt", "") for f in os.listdir(trans_dir)]
    if is_training:
        filenames = [f for f in filenames if f in training]
    else:
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
        #### Resize images
        im = im.resize([int(i) for i in np.floor(np.multiply(resize_to, im.size))])
        return im
    img_list = []
    count = 0
    onepercent = len(data_df)//100
    tenpercent = onepercent*10
    for f in data_df.imgnames:
        img_list.append(readBenthamImg(f))
        count += 1
        if count % onepercent == 0:
            if count % tenpercent == 0:
                perc = count//onepercent
                print(str(perc)+"%", end="", flush=True)
            else:
                print(".", end="", flush=True)
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


    #### Get image sizes and remove big images
    w95 = np.percentile(data_df.widths, 95)
    h95 = np.percentile(data_df.heights, 95)
    data_df = data_df[np.logical_and(data_df.widths < w95, data_df.heights < h95)]
    print("Max image size (width, height): ({0}, {1})".format(w95, h95))
    with open(write_dir + "img_size.txt", "w") as f:
        f.write(",".join([str(w95), str(h95)]))

    #### Find freqency of letters and remove utf-8 special characters
    data_df["nonhex"] = [any([ord(i) > 127 for i in t]) 
                         for t in data_df.transcription]
    data_df = data_df[np.logical_not(data_df.nonhex)]

    letters = dict()
    for tran in data_df.transcription:
        for l in list(tran):
            if l not in letters:
                letters[l] = 0
            letters[l] += 1
    letters = sorted(letters.items(), key = lambda f: f[1], reverse=True)
    with open(write_dir + "alphabet.txt", "w") as f:
        f.write("".join(sorted([l[0] for l in letters])))

    if print_letters:
        print("Letter freqencies:\n", letters)
    else:
        print("Number of letters:", len(letters))

    
    #### Save new images
    if is_training:
        new_img_dir = local_path + "/data/BenthamDataset/Images_mod/"
    else:
        new_img_dir = local_path + "/data/BenthamTest/Images_mod/"
    if not os.path.isdir(new_img_dir):
        os.mkdir(new_img_dir)
    data_df["new_img_path"] = [new_img_dir + f+".png" for f in data_df.filenames]

    print("Saving modified images...")
    count = 0
    onepercent = len(data_df)//100
    tenpercent = onepercent*10
    for i in data_df.index:
        data_df.loc[i, "images"].save(data_df.loc[i, "new_img_path"])
        count += 1
        if count % onepercent == 0:
            if count % tenpercent == 0:
                perc = count//onepercent
                print(str(perc)+"%", end="", flush=True)
            else:
                print(".", end="", flush=True)
    print()
    print(str(count) + " images saved")

    #### Save training file
    export_df = data_df[["new_img_path", "transcription"]]
    export_df.to_csv(write_dir + "train.csv", sep="\t", index=False)
    return export_df


if __name__ == "__main__":
    if len(sys.argv >= 3):
        resize_to = float(sys.argv[2])
    else:
        resize_to = 0.5

    if len(sys.argv >= 4):
        print_letters = sys.argv[3] == "True"
    else:
        print_letters = False

    if len(sys.argv) >= 2:
        if sys.argv[1] == "train":
            preprocess_bentham(True, resize_to, print_letters)
        elif sys.argv[1] == "test":
            preprocess_bentham(False, resize_to, print_letters)
        elif sys.argv[1] == "both":
            preprocess_bentham(True, resize_to, print_letters)
            preprocess_bentham(False, resize_to, print_letters)
        else:
            print("First argument must be one of 'train', 'test', or 'both'")
    else:
        preprocess_bentham(True)
        preprocess_bentham(False)
