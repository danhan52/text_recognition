import numpy as np
import pandas as pd
import os
import sys
import re
from PIL import Image

def preprocess_iam_lines(resize_to=1.0, resize_dir=""):
    #### Read in data
    data_list = []

    with open("../data/iamHandwriting/ascii/lines.txt") as f:
        for line in f:
            if line[0] == "#":
                continue
            line = line.replace("\n", "")
            l_split = line.split(" ", 8)

            data_dict = dict()
            data_dict["lineID"] = l_split[0]
            data_dict["segmentation"] = l_split[1]
            data_dict["bin_thresh"] = int(l_split[2])
            data_dict["n_components"] = int(l_split[3])
            data_dict["x_bound"] = int(l_split[4])
            data_dict["y_bound"] = int(l_split[5])
            data_dict["w_bound"] = int(l_split[6])
            data_dict["h_bound"] = int(l_split[7])
            data_dict["transcription"] = " ".join(l_split[8].split("|"))
            data_list.append(data_dict)

    data_df = pd.DataFrame(data_list)
    data_df = data_df[["lineID", "transcription",
                       "segmentation", "bin_thresh", "x_bound", "y_bound",
                       "w_bound", "h_bound", "n_components"]]


    #### Add new columns
    # location columns
    data_df["prefix"] = [x.split("-")[0] for x in data_df["lineID"]]
    data_df["form"] = ["-".join([x.split("-")[0], x.split("-")[1]])
                                for x in data_df["lineID"]]
    local_path = os.getcwd().replace("\\", "/")
    local_path = local_path.replace("preprocess", "") + "/data/iamHandwriting/lines/"
    data_df["path"] = local_path + data_df["prefix"] + "/" + data_df["form"] + "/" + data_df["lineID"] + ".png"


    #### Get rid of unwanted rows
    w95 = np.percentile(data_df.w_bound, 95)
    h95 = np.percentile(data_df.h_bound, 95)
    print("Max image size (width, height): ({0}, {1})".format(w95, h95))
    with open("../data/iamHandwriting/img_size.txt", "w") as f:
        f.write(",".join([str(w95), str(h95)]))

    # get rid of the really big images
    data_df = data_df[np.logical_and(data_df.w_bound < w95, data_df.h_bound < h95)]
    
    #### Resize images (if requested)
    if resize_to != 1.0:
        if not os.path.isdir("../data/iamHandwriting/" + resize_dir):
            os.mkdir("../data/iamHandwriting/" + resize_dir)
        count = 0
        onepercent = len(data_df)//100
        tenpercent = onepercent*10
        def replace_lines(fn):
            m = re.sub("lines/[a-z]+[0-9]+/[a-z]+[0-9]+-[0-9a-z]+/(.*\.png)", resize_dir+"/\\1", fn)
            m = m.replace("//", "/")
            return m
        
        for fn in data_df["path"]:
            img = Image.open(fn)
            img = img.resize([int(i) for i in np.floor(np.multiply(resize_to, img.size))])
            img.save(replace_lines(fn))
            img.close()
            
            count += 1
            if count % onepercent == 0:
                if count % tenpercent == 0:
                    perc = count//onepercent
                    print(str(perc)+"%", end="", flush=True)
                else:
                    print(".", end="", flush=True)
        data_df["path"] = data_df["path"].apply(replace_lines)
        print("\nResized max image size (width, height): ({0}, {1})".format(str(round(w95*resize_to)), str(round(h95*resize_to))))
        with open("../data/iamHandwriting/img_size.txt", "w") as f:
            f.write(",".join([str(round(w95*resize_to)), str(round(h95*resize_to))]))
    
    #### Save all lines
    data_df["new_img_path"] = data_df["path"]
    data_df = data_df[["new_img_path", "transcription"]]
    data_df.to_csv("../data/iamHandwriting/train.csv", sep="\t", index=False)
    print(str(len(data_df)) + " images in train.csv")
    
    #### Find freqency of letters
    letters = dict()

    for tran in data_df.transcription:
        for l in list(tran):
            if l not in letters:
                letters[l] = 0
            letters[l] += 1
    letters = sorted(letters.items(), key = lambda f: f[1], reverse=True)
    with open("../data/iamHandwriting/alphabet.txt", "w") as f:
        f.write("".join(sorted([l[0] for l in letters])))
    
    print()
    print("Letter freqencies:\n", letters)


def preprocess_iam_words(lower=True):
    #### Read in data
    data_list = []

    with open("../data/iamHandwriting/ascii/words.txt") as f:
        for line in f:
            if line[0] == "#":
                continue
            line = line.replace("\n", "")
            l_split = line.split(" ", 8)

            data_dict = dict()
            data_dict["wordID"] = l_split[0]
            data_dict["segmentation"] = l_split[1]
            data_dict["bin_thresh"] = int(l_split[2])
            data_dict["x_bound"] = int(l_split[3])
            data_dict["y_bound"] = int(l_split[4])
            data_dict["w_bound"] = int(l_split[5])
            data_dict["h_bound"] = int(l_split[6])
            data_dict["grammar"] = l_split[7]
            data_dict["transcription"] = l_split[8]
            data_list.append(data_dict)

    data_df = pd.DataFrame(data_list)
    data_df = data_df[["wordID", "transcription",
                       "segmentation", "bin_thresh", "x_bound", "y_bound",
                       "w_bound", "h_bound", "grammar"]]


    #### Add new columns
    # location columns
    data_df["prefix"] = [x.split("-")[0] for x in data_df["wordID"]]
    data_df["form"] = ["-".join([x.split("-")[0], x.split("-")[1]])
                                for x in data_df["wordID"]]
    local_path = os.getcwd().replace("\\", "/")
    local_path = local_path.replace("preprocess", "") + "/data/iamHandwriting/words/"
    data_df["path"] = local_path + data_df["prefix"] + "/" + data_df["form"] + "/" + data_df["wordID"] + ".png"


    #### Get rid of unwanted rows

    w95 = np.percentile(data_df.w_bound, 95)
    h95 = np.percentile(data_df.h_bound, 95)
    print("Max image size (width, height): ({0}, {1})".format(w95, h95))
    with open("../data/iamHandwriting/img_size.txt", "w") as f:
        f.write(",".join([str(w95), str(h95)]))


    # get rid of the really big images
    data_df = data_df[np.logical_and(data_df.w_bound < w95, data_df.h_bound < h95)]

    # image is broken
    data_df = data_df[data_df.wordID != "r06-022-03-05"]

    # get only words that are entirely lowercase letters
    data_df["lower"] = [all([y.islower() for y in x]) 
                         for x in data_df["transcription"]]
    
    if lower:
        data_df = data_df[data_df["lower"]]


    #### Save words
    data_df["new_img_path"] = data_df["path"]
    data_df = data_df[["new_img_path", "transcription"]]
    data_df.to_csv("../data/iamHandwriting/train.csv", sep="\t", index=False)
    print(str(len(data_df)) + " images in train.csv")

    #### Find freqency of letters
    letters = dict()

    for tran in data_df.transcription:
        for l in list(tran):
            if l not in letters:
                letters[l] = 0
            letters[l] += 1
    letters = sorted(letters.items(), key = lambda f: f[1], reverse=True)
    with open("../data/iamHandwriting/alphabet.txt", "w") as f:
        f.write("".join(sorted([l[0] for l in letters])))
    
    print()
    print("Letter freqencies:\n", letters)



def preprocess_iam():
    if len(sys.argv) < 2:
        print("You must have at least one argument for lines or words")
        sys.exit(0)
    if sys.argv[1] == "lines":
        if len(sys.argv) < 4:
            preprocess_iam_lines()
        else:
            preprocess_iam_lines(float(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "words":
        if len(sys.argv) < 3:
            print("For words, you must specify 0 for only lowercase and 1 for all letters")
            sys.exit(0)
        if int(sys.argv[2]) == 0:
            preprocess_iam_words(True)
        elif int(sys.argv[2]) == 1:
            preprocess_iam_words(False)
        else:
            print("For words, second argument must be 0 (for only lowercase) or 1 (for all letters)")
    else:
        print("First arguments must be either lines or words")
        
if __name__ == "__main__":
    preprocess_iam()