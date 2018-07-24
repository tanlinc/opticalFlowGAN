import os, subprocess
import numpy as np
import scipy, scipy.io
import matplotlib.pyplot as plt

import sys

import OpenEXR

import sys
sys.path.append( '/media/seagate1/StructuredForests' )
import StructuredForests
import skimage, skimage.io

def returnEdgeModel( ):
    options = {
        "rgbd": 0,
        "shrink": 2,
        "n_orient": 4,
        "grd_smooth_rad": 0,
        "grd_norm_rad": 4,
        "reg_smooth_rad": 2,
        "ss_smooth_rad": 8,
        "p_size": 32,
        "g_size": 16,
        "n_cell": 5,

        "n_pos": 10000,
        "n_neg": 10000,
        "fraction": 0.25,
        "n_tree": 8,
        "n_class": 2,
        "min_count": 1,
        "min_child": 8,
        "max_depth": 64,
        "split": "gini",
        "discretize": lambda lbls, n_class:
            discretize(lbls, n_class, n_sample=256, rand=rand),

        "stride": 2,
        "sharpen": 2,
        "n_tree_eval": 4,
        "nms": True,
    }        
    rand = np.random.RandomState(1)
    modelEdge = StructuredForests.StructuredForests(options, rand=rand, model_dir='/media/seagate1/StructuredForests/model')
    return modelEdge


if __name__ == '__main__':

    plt.close('all')

    folder = 'cave_2'
    save_dir = os.path.join('/media/seagate1/SintelResults/', folder )
    if not os.path.exists(save_dir): #'./results/
      os.makedirs(save_dir)

    datapath = '/windows/Data/MPI-Sintel-complete/training/clean'
    fname1 = os.path.join(datapath, folder, 'frame_0016.png')
    fname2 = os.path.join(datapath, folder, 'frame_0017.png')

    edge_file_name = os.path.join(save_dir, 'edge_frame_0016.png' )

    scale = 1.0
    I1 = scipy.misc.imread(fname1) / 255.0
    I2 = scipy.misc.imread(fname2) / 255.0

    I1 = np.minimum(1, np.maximum(0, I1))
    I2 = np.minimum(1, np.maximum(0, I2))

    modelEdge = returnEdgeModel()   
    edge  = modelEdge.predict(I1)
    edgeI = skimage.img_as_ubyte( edge )
    skimage.io.imsave(edge_file_name, edgeI)
    plt.figure(); plt.imshow(edgeI, interpolation='none'); plt.title('edge Image')
    
    plt.show()
