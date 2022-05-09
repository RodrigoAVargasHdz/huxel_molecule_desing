import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

def plot_mosaic():


    # plt.clf()
    # fig, axs = plt.subplots(nrows=4,ncols=2,gridspec_kw = {'wspace':0, 'hspace':0})
    # l = 1 
    # for i, ax in enumerate(fig.axes):
    #     f_molecule = f'molecules/smile{l}_dummyatoms.png'
    #     image = plt.imread(f_molecule)
    #     lum = image[:, :, 0] 
    #     ax.imshow(lum,cmap=cm.gray)
    #     # axs[i,j].axis('off') 
    #     l += 1 
    # plt.savefig('test.png')

    nrow = 4
    ncol = 2
    fig = plt.figure() 

    gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1],
            wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) 

    l = 1
    for i in range(nrow):
        for j in range(ncol):
            f_molecule = f'molecules/smile{l}_dummyatoms.png'
            image = plt.imread(f_molecule)
            ax= plt.subplot(gs[i,j])
            ax.imshow(image)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            l += 1 

    plt.savefig('test.png')
    #plt.tight_layout() # do not use this!!
    # plt.show()


def main():
    plot_mosaic()


if __name__ == "__main__":
    main()