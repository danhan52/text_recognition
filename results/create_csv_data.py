import pickle
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import sys

#if os.path.exists("../modeling/tf_output/online_data/modified_data.csv"):
#    redata = pd.read_csv("../modeling/tf_output/online_data/modified_data.csv")
#else:
def create_csv_data(input_name, output_name, datatype="pred"):
    # Load in data
    # training csv with image locations and transcriptions
    data = pd.read_pickle(input_name)
    pred_data = data[data.pred == datatype]

    # Get pertinent information from only prediction data
    redata = pd.DataFrame(columns=["filename", "label", "word", "subject_id", "class_id", "frame", "line_num", "bunch"])
    for i in range(len(pred_data)):
        curdat = pred_data.iloc[i]
        labels = [str(d, "utf-8") for d in curdat["labels"][0]]
        words = [str(d, "utf-8") for d in curdat["words"][0]]
        filenames = [str(d, "utf-8") for d in curdat["filenames"][0]]

        fns = [f.split("/")[-1].replace(".png", "") for f in filenames]
        ids = [[str(int(float(idd))) for idd in f.split("_")] for f in fns]
        subject_id = [f[0] for f in ids]
        class_id = [f[1] for f in ids]
        frame = [f[2] for f in ids]
        line_num = [f[3] for f in ids]
        bunch_num = [curdat["bunch"] for j in range(len(fns))]

        data_dict = {"filename":filenames, "label":labels, "word":words, 
                     "subject_id":subject_id, "class_id":class_id, "frame":frame,
                     "line_num":line_num, "bunch":bunch_num}
        newdata = pd.DataFrame.from_dict(data_dict, dtype="str")
        redata = redata.append(newdata)

        if i % 100 == 0: print(i, end="\t", flush=True)

    # Figure out which data has been seen before based on subject id
    subj_set = set()
    seen_before = []

    for i in range(len(redata)):
        combo = redata.iloc[i]["subject_id"] + "_" + redata.iloc[i]["frame"] + "_" + redata.iloc[i]["line_num"]
        if combo in subj_set:
            seen_before.append(True)
        else:
            seen_before.append(False)
            subj_set.add(combo)

    redata["unseen_subj"] = np.logical_not(seen_before)

    # get character error rate
    def cer(r, h):
        r = list(r)
        h = list(h)
        # initialisation
        import numpy
        d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion    = d[i][j-1] + 1
                    deletion     = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]/len(r)

    count = 0
    onepercent = len(redata)//100
    tenpercent = onepercent*10

    cers = []
    for i in range(len(redata)):
        cers.append(cer(redata.iloc[i]["label"], redata.iloc[i]["word"]))

        count += 1
        if count % onepercent == 0:
            if count % tenpercent == 0:
                perc = count//onepercent
                print(str(perc)+"%", end="", flush=True)
            else:
                print(".", end="", flush=True)

    redata["cer"] = cers

    redata.to_csv(output_name, index=False)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("You must specify input file name and output file name")
    elif len(sys.argv) == 3:
        create_csv_data(sys.argv[1], sys.argv[2])
    else:
        create_csv_data(sys.argv[1], sys.argv[2], sys.argv[3])
#"../modeling/tf_output/online_data/modified_data.csv"
#"../modeling/tf_output/online_data/online_metrics59000.pkl"