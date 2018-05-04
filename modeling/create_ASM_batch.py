import numpy as np
import pandas as pd
import os
import sys
import pickle

from PIL import Image
import PIL
import requests
from io import BytesIO


def readImg(url, grey=True):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert("L")
    return img

# create a batch with the next group of data from ASM
def create_ASM_batch(batch_end=1000, batch_size=1000, resize_to=1.0, 
                     data_loc="../data", rand_batch=False):
    # Read in all classifications
    sv_fold = data_loc + "/ASM/"
    if not os.path.isdir(sv_fold+"Images"):
        os.mkdir(sv_fold+"Images")
    else:
        for the_file in os.listdir(sv_fold+"Images"):
            fpath = os.path.join(sv_fold+"Images", the_file)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
            except:
                pass

    full_sv = os.getcwd().replace("\\", "/")
    full_sv = full_sv.replace("/online_functions", "")
    full_sv = full_sv.replace("/modeling", "")
    full_sv = full_sv.replace("/results", "")
    full_sv = full_sv + "/data/ASM/Images/"

    print("Loading classification data", flush=True)
    if os.path.exists(data_loc + "/ASM/full_train.csv"):
        data = pd.read_csv(data_loc + "/ASM/full_train.csv", sep="\t")
    else:
        print("File doesn't exist, run \"preprocess_ASM_csv.py\" first", flush=True)

    # Preprocess by splitting image into sections
    # load input_shape from file output by preprocess
    with open(data_loc + "/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        maxh = int(float(h))
        maxw = int(float(w))

    # loop to get data
    if not rand_batch:
        curdata = data.iloc[(batch_end-batch_size):batch_end]
    else:
        curdata = data.iloc[:batch_end]
        curdata = curdata.sample(batch_size) # get random subset
    train_lines = []
    img_files = []
    count = 0
    onepercent = len(curdata)//100
    tenpercent = onepercent*10

    print("Creating image files and training csv", flush=True)
    for i in range(len(curdata)):
        curclas = curdata.iloc[i].copy()
        loc = curclas["location"]
        img = readImg(loc)

        trans = curclas["transcription"]
        # extract area in bounding box
        line_box = eval(curclas["line_box"])
        img_line = img.crop(line_box)

        # resize if it's too big
        if img_line.size[1] > maxh:
            ratio = maxh / float(img_line.size[1])
            wnew = int(float(img_line.size[0]) * float(ratio))
            img_line = img_line.resize((wnew, maxh), PIL.Image.ANTIALIAS)
        if img_line.size[0] > maxw:
            ratio = maxw / float(img_line.size[0])
            hnew = int(float(img_line.size[1]) * float(ratio))
            img_line = img_line.resize((hnew, maxw), PIL.Image.ANTIALIAS)

        
        #img_line = img_line.resize([int(j) for j in np.floor(np.multiply(resize_to, img_line.size))])
        img_line_np = np.array(img_line)

        # turn everything above previous line white - slope top
        m = curclas["slope_top"]
        b = curclas["intercept_top"]
        for x in range(img_line_np.shape[1]):
            img_line_np[:int(np.floor(m*x+b)),x] = 255

        # turn everything below transcription line white - slope bottom
        m = curclas["slope_bottom"]
        b = curclas["intercept_bottom"]
        for x in range(img_line_np.shape[1]):
            img_line_np[int(np.ceil(m*x+b)):,x] = 255

        img_line = Image.fromarray(img_line_np)

        fn = "{0}{1}_{2}_{3}_{4}.png".format(full_sv, curclas["subject_id"], curclas["classification_id"],
                                             curclas["frame"], curclas["j"])
        # save image
        img_line.save(fn)
        img_files.append(fn)
        # add line for training
        train_lines.append(trans)

        count += 1
        if count % onepercent == 0:
            if count % tenpercent == 0:
                perc = count//onepercent
                print(str(perc)+"%", end="", flush=True)
            else:
                print(".", end="", flush=True)

    # end of loop
    savedata = pd.DataFrame.from_dict({"new_img_path":img_files, "transcription":train_lines})
    savedata = savedata[np.logical_not(savedata.duplicated())]
    savedata.to_csv(data_loc + "/ASM/train.csv", sep="\t", index=False)
    print("\nTraining file and {0} images created".format(len(savedata)), flush=True)
    return


if __name__ == "__main__":
    batch_end=1000
    batch_size=1000
    resize_to=1.0
    data_loc="../data"
    rand_batch=False

    if len(sys.argv) >= 6:
        batch_end=int(sys.argv[1])
        batch_size=int(sys.argv[2])
        resize_to=float(sys.argv[3])
        data_loc=sys.argv[4]
        rand_batch=sys.argv[5] == "True"

    redo = True
    while redo:
        try:
            create_ASM_batch(batch_end=batch_end, batch_size=batch_size, 
                             resize_to=resize_to, data_loc=data_loc,
                             rand_batch=rand_batch)
            redo = False
        except:
            print("Error during batch creation, redoing", flush=True)
            redo = True
            