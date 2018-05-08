import numpy as np
import pandas as pd
import os
import sys
import re
from PIL import Image

def preprocess_iam_lines(is_training=True, resize_to=0.5, print_letters=False):
    #### Read in data
    data_list = []
    part_dir = "../data/iamHandwriting/Partitions/"

    # get directory to send all info to
    write_dir = "../data/"
    if is_training:
        write_dir = write_dir + "iamHandwriting/"
    else:
        write_dir = write_dir + "iamTest/"

    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

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

    # get the partitions of the dataset
    with open(os.path.join(part_dir, "trainset.txt")) as f:
        training = f.read().splitlines()
    with open(os.path.join(part_dir, "validationset1.txt")) as f:
        validation1 = f.read().splitlines()
    with open(os.path.join(part_dir, "validationset2.txt")) as f:
        validation2 = f.read().splitlines()
    with open(os.path.join(part_dir, "testset.txt")) as f:
        test = f.read().splitlines()

    if is_training: # do everything that's not test
        data_df = data_df[[l not in test for l in data_df["lineID"]]]
    else: # only do test
        data_df = data_df[[l in test for l in data_df["lineID"]]]

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
    with open(write_dir + "img_size.txt", "w") as f:
        f.write(",".join([str(w95), str(h95)]))

    # get rid of the really big images
    data_df = data_df[np.logical_and(data_df.w_bound < w95, data_df.h_bound < h95)]

    #### Resize images (if requested)
    if resize_to != 1.0 or not is_training:
        resize_dir = "Images_mod/"
        if not os.path.isdir(write_dir + resize_dir):
            os.mkdir(write_dir + resize_dir)
        count = 0
        onepercent = len(data_df)//100
        tenpercent = onepercent*10
        def replace_lines(fn):
            if is_training:
                m = re.sub("lines/[a-z]+[0-9]+/[a-z]+[0-9]+-[0-9a-z]+/(.*\.png)",
                           resize_dir+"/\\1", fn)
            else:
                m = re.sub("iamHandwriting/lines/[a-z]+[0-9]+/[a-z]+[0-9]+-[0-9a-z]+/(.*\.png)",
                           "iamTest/"+resize_dir+"/\\1", fn)
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
        with open(write_dir + "img_size.txt", "w") as f:
            f.write(",".join([str(round(w95*resize_to)), str(round(h95*resize_to))]))

    #### Save all lines
    data_df["new_img_path"] = data_df["path"]
    data_df = data_df[["new_img_path", "transcription"]]
    data_df.to_csv(write_dir + "train.csv", sep="\t", index=False)
    print(str(len(data_df)) + " images in train.csv")

    #### Find freqency of letters
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
    return


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
            preprocess_iam_lines(True, resize_to, print_letters)
        elif sys.argv[1] == "test":
            preprocess_iam_lines(False, resize_to, print_letters)
        elif sys.argv[1] == "both":
            preprocess_iam_lines(True, resize_to, print_letters)
            preprocess_iam_lines(False, resize_to, print_letters)
        else:
            print("First argument must be one of 'train', 'test', or 'both'")
    else:
        preprocess_iam_lines(True)
        preprocess_iam_lines(False)
