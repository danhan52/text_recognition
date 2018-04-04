import numpy as np
import pandas as pd
import os
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

def create_ASM_batch(batch_start=0, batch_end=100):
    # Read in all classifications
    sv_fold = "../../data/ASM/"
    if not os.path.isdir(sv_fold+"Images"):
        os.mkdir(sv_fold+"Images")

    full_sv = os.getcwd().replace("\\", "/")
    full_sv = full_sv.replace("/online_functions", "")
    full_sv = full_sv.replace("/modeling", "")
    full_sv = full_sv + "/data/ASM/Images/"\

    print("Loading classification data")
    if os.path.exists("../../data/ASM/newclas.pkl"):
        data = pickle.load(open("../../data/ASM/newclas.pkl", "rb"))
        print("\n{0} data entries loaded".format(len(data)))
    else:
        print("File doesn't exist, run \"preprocess_ASM_csv.py\" first")


    # Preprocess by splitting image into sections
    # load input_shape from file output by preprocess
    with open("../../data/combined/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        maxh = int(float(h))

    # loop to get data
    curdata = data.iloc[batch_start:batch_end]
    train_lines = []
    count = 0

    print("Creating image files and training csv")
    for i in range(len(curdata)):
        curclas = curdata.iloc[i].copy()
        loc = curclas["location"]

        img = readImg(loc)
        pts = curclas["annodata"]

        for j in range(len(pts)):
            curpts = pts.iloc[j]
            trans = curpts["transcription"]
            # skip all uncertain transcriptions
            if "[unclear]" in trans:
                continue
            # extract area in bounding box
            line_box = (int(curpts["x1"]), # x
                        int(min(curpts["y3"],curpts["y4"])), # y 
                        int(curpts["x2"]), # w
                        int(max(curpts["y1"],curpts["y2"]))) # h
            img_line = img.crop(line_box)
            if img_line.size[0] <= 0 or img_line.size[1] <= 0:
                continue

            # resize if it's too big
            if img_line.size[1] > maxh:
                ratio = maxh / float(img.size[1])
                wnew = int(float(img.size[0]) * float(ratio))
                img_line.resize((wnew, maxh), PIL.Image.ANTIALIAS)

            img_line_np = np.array(img_line)

            # turn everything above previous line white
            m = (curpts["y4"]-curpts["y3"])/(curpts["x2"]-curpts["x1"])
            b = curpts["y3"]-min(curpts["y4"],curpts["y3"])
            for x in range(img_line_np.shape[1]):
                img_line_np[:int(np.floor(m*x+b)),x] = 255

            # turn everything below transcription line white
            m = (curpts["y2"]-curpts["y1"])/(curpts["x2"]-curpts["x1"])
            b = curpts["y1"]-min(curpts["y4"],curpts["y3"])
            for x in range(img_line_np.shape[1]):
                img_line_np[int(np.ceil(m*x+b)):,x] = 255

            img_line = Image.fromarray(img_line_np)

            fn = "{0}{1}_{2}_{3}_{4}.png".format(full_sv, curclas["subject_id"], curclas["classification_id"], curclas["frame"], j)
            # save image
            img_line.save(fn)
            # add line for training
            train_lines.append(fn + "\t" + trans + "\n")

        count += 1
        if count % ((batch_end-batch_start)/10) == 0: print(count, end="\t", flush=True)
    # end of loop
    with open("../../data/ASM/train.csv", "w") as f:
        for l in train_lines:
            f.write(l)
    print("Images and training file created")

        
if __name__ == "__main__":
    create_ASM_batch()