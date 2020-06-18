"""
==========================
Creating adjacent subplots
==========================

"""


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2



fig, ax = plt.subplots(5, 4, figsize=(5, 4))
fig.subplots_adjust(hspace=0, wspace=0)

core=0
for i in range(5):
    for j in range(4):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        im = cv2.imread('core'+str(core)+'.png',0)
        ax[i, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[i,j].set_aspect('auto')
        core+=1

plt.savefig("cores_sim.pdf", bbox_inches='tight')
plt.show()

