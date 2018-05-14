from scipy.signal import argrelextrema, find_peaks_cwt
from skimage.feature import canny
from operator import itemgetter
from itertools import groupby
import numpy as np

# project the image onto a specific direction
def project(img, direction):
    if direction == "x":
        proj = np.sum(img, 0)
    elif direction == "y":
        proj = np.sum(img, 1)
    else:
        print("Direction must be one of 'x' or 'y'")
        proj = []
    return proj
    
# get breaks based on projection
def projBreaks(img, direction, thresh = 0):
    proj = project(img, direction)
    breaks = argrelextrema(proj, np.greater_equal)[0]
    # breaks = find_peaks_cwt(vector=proj, widths=np.arange(*(5, 60)))
    breaks = breaks[breaks > thresh-1]
    # append 0 and image size to ends of vector
    if not breaks[0] == 0:
        breaks = np.insert(breaks, 0, 0)
    if not breaks[-1] == proj.shape[0]:
        breaks = np.append(breaks, proj.shape[0]) 
    return breaks.astype("int")

def filterBreaks(lb_old, matchlim = 30):
    lb = []
    lb.append(lb_old[0])
    cur = 0
    for i in range(1, len(lb_old)):
        if np.abs(lb[cur] - lb_old[i]) < matchlim:
            lb[cur] = np.mean([lb[cur], lb_old[i]])
        else:
            cur += 1
            lb.append(lb_old[i])
    lb = np.array(lb).astype("int")
    return lb

