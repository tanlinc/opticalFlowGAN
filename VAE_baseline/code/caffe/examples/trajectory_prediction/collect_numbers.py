#!/usr/bin/python
#execfile('myaddpath.py');
import caffe
from scipy import ndimage
from scipy import misc
from scipy import signal
from scipy import io
import h5py as hpy;
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import os
import csv,math
import subprocess, sys
import string  
import lmdb
import random
import heapq
from multiprocessing import Process, Queue, Pipe
import time;
from mako.template import Template
from mako import exceptions
import traceback
import skimage.transform
import sklearn.cluster
import sklearn.decomposition as ml
import pickle
from PIL import Image, ImageDraw;
import scipy.fftpack as fft
import math
import colorsys;
from scipy import ndimage;

#outdir = "/nfs/hn48/jcwalker/quantitative_results/";
#outdir = "/nfs/hn48/jcwalker/optical_flow/";
outdir = "/nfs/ladoga_no_backups/users/jcwalker/doWork/";
im_sz = (256,320);

num_samples = 50;

g = open(outdir + "/imagelist.txt", "r");

final_scores = [];
final_names = [];
eucl_dists = [];

for roundid in range(0,23):
  line = g.readline();

  columns = line.split(",")
  columns = [col.strip() for col in columns]

  im_name = columns[0];
  label_name = columns[1];
  print roundid
  (blah1, name) = os.path.split(im_name);
  sampledir = outdir + "/" + name + "_samples/";

  (eucl_distance) = pickle.load(open(sampledir + "/thescores.py", "r"))

  eucl_dists.append(eucl_distance);


eucl_dists = np.array(eucl_dists);
print(eucl_dists.shape)
the_min_scores = [];
for i in range(1,num_samples):
	themins = np.min(eucl_dists[:,0:i],axis=1);
	the_min_scores.append(np.mean(themins));
the_min_scores = np.array(the_min_scores);
print(the_min_scores)




