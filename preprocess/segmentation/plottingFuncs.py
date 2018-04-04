import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle as Rec

# plot "histogram"
def densityHist(proj, pers = []):
    plt.plot(proj)
    ny = proj.shape[0]
    for pe in pers:
        p = np.percentile(proj, pe)
        plt.plot([0,ny], [p, p])
        plt.text(x=ny, y=p, s=str(pe))
    plt.show()

# plot boxes around words (or just linebreaks)
def plotBoxes(img, lb, wb = [], cb = [], cmap = None, saveFile = None):
    if len(lb) == 0:
        lb = [0, img.shape[0]]
    plt.imshow(img, cmap=cmap)
    plt.plot([0,img.shape[1]], [lb, lb], 'b')
    if len(wb) > 0:
        for i in range(len(lb)-1):
            plt.plot([wb[i], wb[i]], [lb[i], lb[i+1]], 'b')
            if len(cb) > 0:
                for j in range(len(wb[i])-1):
                    plt.plot([np.add(wb[i][j], cb[i][j]), np.add(wb[i][j],cb[i][j])],
                             [lb[i], lb[i+1]], 'r')
    if saveFile is not None:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1,bottom=0,right=1,left=0,
                            hspace=0,wspace=0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
        plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
        plt.savefig(saveFile, bbox_inches="tight",
                    pad_inches=0)
        plt.close()
    else:
        plt.show()
    return

# just a shortcut for plotting
def plquick(img, cmap="nipy_spectral", figsize=None):
    if figsize is None:
        plt.imshow(img, cmap=cmap)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, cmap=cmap)
    plt.show()
    