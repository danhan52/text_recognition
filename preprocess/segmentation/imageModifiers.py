import numpy as np
import matplotlib.pyplot as plt

from skimage import filters as skimfilt
from scipy.ndimage import label as sci_lab

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

# binarize an image
def binarizeImg(img, threshFn = None, biThresh = None, greater = True, plotIt = False):
    if threshFn is not None:
        biThresh = threshFn(img)
    elif biThresh is None:
        biThresh = 0
    if greater:
        imgCp = img > biThresh
    else:
        imgCp = img < biThresh
    
    if plotIt:
        plt.imshow(imgCp, cmap = "gray")
        plt.show()
    return imgCp, biThresh

# smooth an image
def smoothImg(img, sigma, plotIt = False):
    imgCp = skimfilt.gaussian(img, sigma=sigma, multichannel=False)
    if plotIt:
        plt.imshow(imgCp, cmap = "gray")
        plt.show()
    return imgCp

# remove the edges from an image
def removeEdges(grey, let, pageBlur, plotit = False):
    level1Mask = binarizeImg(grey, skimfilt.threshold_triangle, greater=False)[0]
    blurredLevel1Mask = smoothImg(level1Mask, sigma=pageBlur)

    # only do the "soft" trim from Hugh's code
    level2Mask = binarizeImg(blurredLevel1Mask, skimfilt.threshold_mean, greater=False)[0]
    xProjectedL2Mask = project(level2Mask, 'y')
    yProjectedL2Mask = project(level2Mask, 'x')
    level2TrimMask = np.outer(xProjectedL2Mask, yProjectedL2Mask) > 0

    # find the blob in the center: should be the letter
    labels, nrObj = sci_lab(level2TrimMask)
    nr, nc = grey.shape
    center_label = labels[int(nr/2.0), int(nc/2.0)]
    level2TrimMask[labels != center_label] = 0

    # reproject with just letter
    xProj2 = project(level2TrimMask, 'y')
    yProj2 = project(level2TrimMask, 'x')

    level2TrimOffsets = (np.nonzero(xProj2)[0][0], 
                         np.nonzero(yProj2)[0][0])

    x_count = np.count_nonzero(xProj2)
    y_count = np.count_nonzero(yProj2)
    greyTrimmed = grey[level2TrimMask].reshape(x_count, y_count)
    letTrimmed = let[level2TrimMask].reshape(x_count, y_count, 3)
    
    if plotit:
        # plot level 1
        subjectFigure, subjectAxes = plt.subplots(figsize=(30, 20),
                                                      ncols=2, nrows=1)
        subjectAxes.flatten()[0].imshow(level1Mask, cmap = 'gray')
        subjectAxes.flatten()[1].imshow(blurredLevel1Mask, cmap = 'gray')
        plt.show()

        # plot level 2
        subjectFigure, subjectAxes = plt.subplots(figsize=(30, 20), ncols=3, nrows=2)
        subjectAxes.flatten()[0].imshow(level2Mask, cmap = 'gray')
        subjectAxes.flatten()[1].imshow(level2TrimMask, cmap = 'gray')
        subjectAxes.flatten()[2].imshow(greyTrimmed, cmap = 'gray')
        plt.show()
        
    return greyTrimmed, letTrimmed, level2TrimOffsets
