import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
from datetime import datetime

from segmentation.imageModifiers import *
from segmentation.plottingFuncs import *
from segmentation.projEdgeBreaks import *


def preprocess_ASM_csv():
    # Read in all classifications
    sv_fold = "../data/ASM/"
    if not os.path.isdir(sv_fold+"Images"):
        os.mkdir(sv_fold+"Images")

    full_sv = os.getcwd().replace("\\", "/")
    full_sv = full_sv + "data/ASM/Images/"

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


    # Change json columns to dictionaries
    print("Converting json to dictionaries")
    subj["metadata_dict"] = [json.loads(q) for q in subj["metadata"]]
    subj["locations_dict"] = [json.loads(q) for q in subj["locations"]]

    clas["metadata_dict"] = [json.loads(q) for q in clas["metadata"]]
    clas["annotations_dict"] = [json.loads(q) for q in clas["annotations"]]
    clas["subject_data_dict"] = [json.loads(q) for q in clas["subject_data"]]


    # Reorganize data so I only have the relevant information
    # load input_shape from file output by preprocess
    with open("../data/combined/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        maxh = int(float(h))

    def get_transcription_lines(pts, maxh=maxh):
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
            return data_mini

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
            y3 = [max(data_mini["y1"].iloc[0]-maxh, 0)]
            y4 = [max(data_mini["y2"].iloc[0]-maxh, 0)]

        data_mini["y3"] = y3
        data_mini["y4"] = y4

        return data_mini


    # The main loop for creating the new data frame
    data = pd.DataFrame(columns=["subject_id", "classification_id", "workflow_id", "frame", "created_dt", 
                                 "annotations", "metadata_clas", "metadata_subj", "location", "annodata"])

    print("Processing", len(clas), "items...")
    count = 0
    for i in range(len(clas)):
        clas1 = clas.iloc[i]
        subj1 = subj[subj["subject_id"] == clas1["subject_ids"]]

        val = clas1["annotations_dict"][0]["value"]
        frames = np.sort(np.unique([v["frame"] for v in val]))

        for fr in frames:
            fr = frames[0]
            pts = [v for v in val if v["frame"] == fr]

            data_mini = get_transcription_lines(pts)

            newdata = {"subject_id": clas1["subject_ids"],
                       "classification_id": clas1["classification_id"],
                       "workflow_id": clas1["workflow_id"],
                       "frame": fr,
                       "created_dt": clas1["created_dt"],
                       "annotations": clas1["annotations"],
                       "metadata_clas": [clas1["metadata_dict"]],
                       "metadata_subj": [subj1["metadata_dict"]],
                       "location": subj1["locations_dict"].iloc[0][str(fr)],
                       "annodata": [data_mini]}
            newdata = pd.DataFrame.from_dict(newdata)
            data = data.append(newdata)
        count += 1
        if count % 1000 == 0: print(count, end="\t", flush=True)

    # end of loop
    data.sort_values("created_dt")
    pickle.dump(data, open("../data/ASM/newclas.pkl", "wb"))
    print("\nCreated {0} data entries and saved file".format(len(data)))

if __name__ == "__main__":
    preprocess_ASM_csv()