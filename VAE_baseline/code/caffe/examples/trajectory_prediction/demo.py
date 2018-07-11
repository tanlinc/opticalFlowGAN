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

def makeAndSaveSamples(thesamples,outdir,im_name,num_clusters, im_sz):
  im=misc.imread(im_name);
  im = misc.imresize(im, im_sz);
  (blah1, name) = os.path.split(im_name);
  sampledir = outdir + "/" + name + "_samples/";
  print sampledir
  try:
	os.mkdir(sampledir);
  except:
	print "oops sampledir"


  thesamples = np.array(thesamples);
  numbatches = thesamples.shape[0]+1
  thesamples = np.reshape(thesamples,((numbatches-1)*16,10*64*80));
  themeans = [];

  for i in range(0,thesamples.shape[0]):
	themeans.append(np.mean(np.abs(thesamples[i,:])))

  themeans = np.array(themeans);
  themeans = np.percentile(themeans,50);
  temp = [];

  masktemp = [];
  for i in range(0,thesamples.shape[0]):
	if(np.mean(np.abs(thesamples[i,:])) > themeans):
		temp.append(thesamples[i,:]);
  thesamples = np.array(temp);

  for i in range(0,thesamples.shape[0]):
	masktemp.append(thesamples[i,:]/np.linalg.norm(thesamples[i,:]));

  masktemp = np.array(masktemp);
  themask = np.sum(np.abs(masktemp), axis=0);
  themask = np.reshape(themask, (10, 64,80));
  themask = np.sum(np.abs(themask), axis=0);

  themask = themask/(np.max(themask) - np.min(themask));

  kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters);
  kmeans.fit((thesamples));
  J = kmeans.predict(thesamples);
  theinds = np.argsort(-np.bincount(J))

  for curstep in range(1,(num_clusters+1)):
	t = kmeans.cluster_centers_[theinds[curstep - 1],:];
	t = np.reshape(t, (1,10,64,80));
	for i in range(0,10):
		t[0,i,:,:] = np.multiply(t[0,i,:,:], themask)
	ut.drawPixels(im,t,sampledir, curstep, True);
	ut.drawArrows(im,t,sampledir, curstep, True);

  np.save(sampledir + "/thesamples.py", thesamples);
  np.save(sampledir + "/themask.py", themask);
  sortinds = -np.sort(-np.bincount(J));
  np.save(sampledir + "/sortinds.py", sortinds);
  pickle.dump(kmeans,open(sampledir + "/kmeans.py", "wb"))

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

outdir = "./";
im_sz = (256,320);

num_samples = 50;
try:
	os.mkdir(outdir);
except:
	print "oops outdir"
f = open(outdir + "/imagelist.txt", "r");

caffe.set_mode_cpu();
#caffe.set_device(0);
net = caffe.Net(outdir + 'network_fine.prototxt', outdir + '/final.caffemodel', caffe.TEST);

for roundid in range(0,2):
  im_name = f.readline();
  im_name = im_name.rstrip();
  im=misc.imread(im_name);
  im = misc.imresize(im, im_sz);
  im_data = prep_image(im,im_sz).transpose(2,0,1);
  start=time.time();
  net.blobs['imdata'].data[:]=(im_data[...]);
  thesamples = [];

  for batch in range(0,num_samples):
  	samplezs(net)
  	net.forward();
	print batch
	#p = net.blobs["conv2reconst_exp"].data;
	p = net.blobs["conv2reconst"].data;
	u = net.blobs["deconv3reconst"].data;
	
	pp = np.zeros([16,10,64,80], np.float32);
        #u = np.zeros([16,10,64,80], np.float32);
	
	for j in range(0,10):
		for i in range(0,16):
			pp[i,j,:,:] = ndimage.interpolation.zoom(np.reshape(p[i,j,:,:],(16,20)), 4.0, order=0)
        #                u[i,j,:,:] = ndimage.interpolation.zoom(np.reshape(t[i,j,:,:],(16,20)), 4.0, order=0)

	thesamples.append(np.multiply(u,pp));
  print time.time()-start

  makeAndSaveSamples(thesamples,outdir,im_name,5,im_sz);




