import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
from datetime import datetime

def read_subjects_and_classifications(sv_fold):
    print("Reading in subject and classification files")
    subjFile = sv_fold + "anti-slavery-manuscripts-subjects.csv"
    subj = pd.read_csv(subjFile, low_memory=False)

    classFile = sv_fold + "anti-slavery-manuscripts-classifications.csv"
    clas = pd.read_csv(classFile, low_memory=False)

    # Filter out records that haven't been classified or are in the wrong workflow
    subj = subj[subj.classifications_count > 0]
    subj = subj[subj["subject_set_id"] == 15582]
    subj = subj[subj["workflow_id"] == 5329]
    clas = clas[clas.subject_ids.isin(subj.subject_id)]

    # remove everything before 2018 january 23 (full launch)
    dt_frmt = "%Y-%m-%d  %H:%M:%S UTC"
    clas["created_dt"] = [datetime.strptime(i, dt_frmt) for i in clas["created_at"]]
    clas = clas[clas["created_dt"] > datetime(2018, 1, 22)]
    subj = subj[subj["subject_id"].isin(clas["subject_ids"])]
    clas = clas.sort_values("created_dt")

    # Change json columns to dictionaries
    print("Converting json to dictionaries")
    subj["metadata_dict"] = [json.loads(q) for q in subj["metadata"]]
    subj["locations_dict"] = [json.loads(q) for q in subj["locations"]]

    clas["metadata_dict"] = [json.loads(q) for q in clas["metadata"]]
    clas["annotations_dict"] = [json.loads(q) for q in clas["annotations"]]
    clas["subject_data_dict"] = [json.loads(q) for q in clas["subject_data"]]
    
    return subj, clas

def get_transcription_lines(pts):
    with open("../data/combined/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        maxh = int(float(h))
        
    with open("../data/combined/alphabet.txt", "r") as f:
        alphabet = f.readline()
    
    data_mini = pd.DataFrame(columns=["transcription", "x1", "x2", "y1", "y2"])

    # get basic x and y information and make sure x is before y
    for p in pts:
        trans = p["details"][0]["value"]
        xy1 = (p["points"][0]["x"], p["points"][0]["y"])
        xy2 = (p["points"][1]["x"], p["points"][1]["y"])
        xy = [xy1, xy2]
        xy.sort()
        newdata1 = {"transcription": trans,
                   "x1": xy[0][0], "x2": xy[1][0],
                   "y1": xy[0][1], "y2": xy[1][1]}
        newdata1 = pd.DataFrame.from_records([newdata1])
        data_mini = data_mini.append(newdata1)

    # ignore all cross writing or weird lines 
    # i.e. everything where $\Delta x < \Delta$y
    good_lines = abs(data_mini["x1"]-data_mini["x2"]) > abs(data_mini["y1"]-data_mini["y2"])
    data_mini = data_mini[good_lines]
    data_mini = data_mini.sort_values("y1")

    if len(data_mini) <= 0:
        return None

    # get bounding boxes using previous line's y information
    # for first line, use average line height
    if len(data_mini) > 1:
        y3 = list(data_mini["y1"].copy())
        y3_ave = np.max(np.abs(np.diff(y3)))
        y3 = y3[:-1]
        y3.insert(0, max(y3[0]-y3_ave, 0))

        y4 = list(data_mini["y2"].copy())
        y4_ave = np.max(np.abs(np.diff(y4)))
        y4 = y4[:-1]
        y4.insert(0, max(y4[0]-y4_ave, 0))
    else: # if there's only one line, guess based on my model's max height
        ychoice = max(data_mini["y1"].iloc[0], data_mini["y2"].iloc[0])
        y3 = [max(ychoice-maxh, 0)]
        y4 = [max(ychoice-maxh, 0)]

    data_mini["y3"] = y3
    data_mini["y4"] = y4

    # create boxes for sections of image to crop
    def line_box(curpts):
        lb = (int(curpts["x1"]), # x1
              int(min(curpts["y3"],curpts["y4"])), # y1
              int(curpts["x2"]), # x2
              int(max(curpts["y1"],curpts["y2"]))) # y2
        return lb
    data_mini["line_box"] = data_mini.apply(line_box, axis=1)

    # get the slope to use when whiting out the image during preprocessing
    def slope_top(curpts):
        return (curpts["y4"]-curpts["y3"])/(curpts["x2"]-curpts["x1"])
    def intercept_top(curpts):
        return curpts["y3"]-min(curpts["y4"],curpts["y3"])
    def slope_bottom(curpts):
        return (curpts["y2"]-curpts["y1"])/(curpts["x2"]-curpts["x1"])
    def intercept_bottom(curpts):
        return curpts["y1"]-min(curpts["y4"],curpts["y3"])
    data_mini["slope_top"] = data_mini.apply(slope_top, axis=1)
    data_mini["intercept_top"] = data_mini.apply(intercept_top, axis=1)
    data_mini["slope_bottom"] = data_mini.apply(slope_bottom, axis=1)
    data_mini["intercept_bottom"] = data_mini.apply(intercept_bottom, axis=1)

    # get rid of lines that are likely incorrect based on dims of bounding box
    widths = [abs(lb[0] - lb[2]) for lb in data_mini["line_box"]]
    heights = [abs(lb[1] - lb[3]) for lb in data_mini["line_box"]]
    bad_width = [w <= 10 for w in widths]
    bad_height = [h <= 10 for h in heights]
    bad_ratio = [heights[i] > 0.9*widths[i] for i in range(len(widths))]

    # get rid of lines that have transcripts that I don't like
    def bad_trans_fn(curpts):
        t = curpts["transcription"]
        meta = "[unclear]" in t or "[underline]" in t or "[deletion]" in t
        alpha = any([l not in alphabet for l in t])
        return meta or alpha
    bad_trans = data_mini.apply(bad_trans_fn, axis=1)

    bad_lines = np.logical_or(np.logical_or(bad_width, bad_height),
                               np.logical_or(bad_ratio, bad_trans))
    data_mini = data_mini[np.logical_not(bad_lines)]
    data_mini = data_mini.sort_values("y1")
    data_mini = data_mini.drop(["x1", "x2", "y2", "y3", "y4"], axis=1)
    data_mini["j"] = list(range(len(data_mini)))
        
    return data_mini

def preprocess_ASM_csv():
    # Read in all classifications
    sv_fold = "../data/ASM/"
    if not os.path.isdir(sv_fold+"Images"):
        os.mkdir(sv_fold+"Images")

    full_sv = os.getcwd().replace("\\", "/")
    full_sv = full_sv + "data/ASM/Images/"

    subj, clas = read_subjects_and_classifications(sv_fold)

    # Reorganize data so I only have the relevant information
    # The main loop for creating the new data frame
    data = pd.DataFrame(columns=["subject_id", "classification_id", "workflow_id", "frame",
                                 "created_dt", "location", "transcription", "line_box",
                                 "y1", "j", "slope_top", "intercept_top", "slope_bottom",
                                 "intercept_bottom"])

    print("Processing", len(clas), "classifications...")
    # for displaying progress
    count = 0
    onepercent = len(clas)//100
    tenpercent = onepercent*10

    for i in range(len(clas)):
        clas1 = clas.iloc[i]
        subj1 = subj[subj["subject_id"] == clas1["subject_ids"]]

        val = clas1["annotations_dict"][0]["value"]
        frames = np.sort(np.unique([v["frame"] for v in val]))

        for fr in frames:
            pts = [v for v in val if v["frame"] == fr]
            # get data frame of x and y information
            data_mini = get_transcription_lines(pts)
            if data_mini is None:
                continue

            # add data that's constant for the whole classification
            d_loop = range(len(data_mini))
            data_mini["subject_id"] = [clas1["subject_ids"] for i in d_loop]
            data_mini["classification_id"] = [clas1["classification_id"] for i in d_loop]
            data_mini["workflow_id"] = [clas1["workflow_id"] for i in d_loop]
            data_mini["frame"] = [fr for i in d_loop]
            data_mini["created_dt"] = [clas1["created_dt"] for i in d_loop]
            data_mini["location"] = [subj1["locations_dict"].iloc[0][str(fr)] for i in d_loop]

            data = data.append(data_mini)
        count += 1
        if count % onepercent == 0:
            if count % tenpercent == 0:
                perc = count//onepercent
                print(str(perc)+"%", end="", flush=True)
                data.to_csv("../data/ASM/full_train.csv", sep="\t", index=False)
            else:
                print(".", end="", flush=True)
            

    # end of loop
    data.sort_values(["created_dt", "classification_id", "frame", "y1"])
    data.to_csv("../data/ASM/full_train.csv", sep="\t", index=False)
    print("\nCreated {0} data entries and saved file".format(len(data)))

    # create file for length of data
    with open("../data/ASM/data_size.txt", "w") as f:
        f.write(str(len(data)))
    # create img size
    with open("../data/combined/img_size.txt", "r") as f1:
        with open("../data/ASM/img_size.txt", "w") as f2:
            f2.write(f1.readline())
    # create alphabet
    with open("../data/combined/alphabet.txt", "r") as f1:
        with open("../data/ASM/alphabet.txt", "w") as f2:
            f2.write(f1.readline())

if __name__ == "__main__":
    preprocess_ASM_csv()