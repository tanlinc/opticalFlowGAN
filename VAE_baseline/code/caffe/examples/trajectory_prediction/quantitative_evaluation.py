# -*- coding: utf-8 -*-
#   Copyright (C) 2016, Jacob Walker, Carl Doersch.
#   Based on code originally written by Carl Doersch.
#   This code is subject to the (new) BSD license:
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# If this software is used please cite the following paper:
#  An Uncertain Future: Forecasting from Static Images using Variational Autoencoders
#  Jacob Walker, Carl Doersch, Abhinav Gupta, Martial Hebert
#  ECCV 2016


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
import os
import csv,math
import subprocess, sys
import string  
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
import shutil

def sampleEvaluation(samples, ground_truth):
	eucl_distance = [];
	thesamples = np.reshape(samples, (samples.shape[0],10*16*20));
	for i in range(0,samples.shape[0]):
		thediff = samples[i,:,:,:] -  ground_truth[:,:,:];
		thediff = np.multiply(thediff,thediff)
		thediff = np.sum(thediff, axis=0);	
		eucl_distance.append(np.mean(thediff));

	return eucl_distance

def evaluateSamples(thesamples,outdir,im_name,num_clusters, im_sz, label_name):
  im=misc.imread(im_name);
  im = misc.imresize(im, im_sz);
  (blah1, name) = os.path.split(im_name);
  sampledir = outdir + "/" + name + "_samples/";
  print sampledir
  try:
        os.mkdir(sampledir);
  except:
        print "oops"


  f=hpy.File(label_name, 'r');
  trajs = f["trajs"][()]
  trajs = np.concatenate((trajs[0,:,:,:], trajs[1,:,:,:]), axis = 0);
  trajs = trajs.transpose(0,2,1);
  f.close();
  trajs = skimage.transform.downscale_local_mean(trajs, (1,16,16))

  (eucl_distance) = sampleEvaluation(thesamples/1000.0, trajs);

  print (eucl_distance)

  pickle.dump((eucl_distance),open(sampledir + "/thescores.py", "wb"))

  np.save(sampledir + "/thesamples.py", thesamples);
  np.save(sampledir + "/thegt.py", trajs);
  misc.imsave(sampledir + "/" + name, im);
  pickle.dump((im_name, label_name),open(sampledir + "/thenames.py", "wb"))


def samplezs(net):
  for i in range(0,16):
    for j in range(0,2):
      net.blobs["themagzsample"].data[i,j,0,0] = random.gauss(0,1.0);

  for i in range(0,16):
    for j in range(0,8):
      net.blobs["thezsample"].data[i,j,:,:] = random.gauss(0,1.0);

def prep_image(im,origsz):
  im=im.astype(np.float32);
  mean=[123,117,104];
  im=im[:,:,[2,1,0]]
  im2=np.zeros((im.shape[0],im.shape[1],im.shape[2]));
  for i in range(0,3):
      im2[:,:,i]=im[:,:,i]-mean[i]
  im2=im2
  return im2;

outdir = "/nfs/ladoga_no_backups/users/jcwalker/doWork/";
im_sz = (256,320);

num_samples = 50;
try:
	os.mkdir(outdir);
except:
	print "oops"
f = open(outdir + "/imagelist.txt", "r");

caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(outdir + 'network.prototxt', outdir + '/final.caffemodel', caffe.TEST);

for roundid in range(0,10000):
  line = f.readline();

  columns = line.split(",")
  columns = [col.strip() for col in columns]

  im_name = columns[0];
  label_name = columns[1];

  im=misc.imread(im_name);
  im = misc.imresize(im, im_sz);
  im_data = prep_image(im,im_sz).transpose(2,0,1);
  start=time.time();
  net.blobs['imdata'].data[:]=(im_data[...]);
  thesamples = [];
  print roundid
  for batch in range(0,num_samples):
  	samplezs(net)
  	net.forward();
	print batch
	p = net.blobs["conv2reconst"].data;
	t = net.blobs["conv1reconst"].data;
	thesamples.append(np.multiply(p,t));
  print time.time()-start

  thesamples = np.array(thesamples);
  thesamples = np.reshape(thesamples,((thesamples.shape[0])*16,10,16,20));

  themags = np.reshape(thesamples,((thesamples.shape[0]),10*16*20));
  themags = np.multiply(themags, themags);
  themags = np.sum(themags, axis=1);
  theinds = np.argsort(-themags);
  
  evaluateSamples(thesamples,outdir,im_name,5,im_sz, label_name);




