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
#
import inspect
import numpy as np;
import code
import sys
import re
import h5py as hpy;
import sklearn.cluster
import os
from sklearn.neighbors import KernelDensity
if os.environ.get('DISPLAY'):
  import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
import socket
from scipy import misc
import skimage.transform
import scipy.stats as stats
import scipy.stats.kde
import statsmodels.nonparametric.kernel_density as KD;
import numpy.matlib
import math
import copy
import heapq
#import caffe
import random
import traceback
import scipy
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import lmdb
from PIL import Image, ImageDraw;
import scipy.fftpack as fft
import math
from images2gif import writeGif
import colorsys;
from scipy import ndimage;
import sklearn.mixture

def mfilename():
  stack = inspect.stack();
  filepath = (stack[1][1]);
  dotidx=[m.start() for m in re.finditer('\.', filepath)]
  filenm = filepath[0:dotidx[len(dotidx)-1]];
  return filenm;

def makeLmdb(filename, outputFilename):
	g = file(filename)
        cnt = 0;
	supercnt = 0;
	X = np.zeros((3,256,320), np.uint8);

	f = file(filename)
	imgFilenames = []
	trajFilenames = []
        cnt = 0;
	for line in f:
                cnt = cnt + 1;
		columns = line.split(",")
		columns = [col.strip() for col in columns]
                print cnt
		imgFilenames.append(columns[0])              
		trajFilenames.append(columns[-1]);
        cnt = 0;
	imindex = 0;
	env = lmdb.open(outputFilename, map_size=X.nbytes*len(imgFilenames)*2);
	inds = np.array(range(len(imgFilenames)));
        np.random.seed(0);
	np.random.shuffle(inds);
	for i in range(len(imgFilenames)):
		print imindex
		im=misc.imread(imgFilenames[inds[imindex]]);
                im = misc.imresize(im, (256,320));
		f=hpy.File(trajFilenames[inds[imindex]], 'r');
  		trajs = f["trajs"][()]
  		trajs = np.concatenate((trajs[0,:,:,:], trajs[1,:,:,:]), axis = 0);
  		trajs = trajs.transpose(2,1,0);
  		f.close();
		trajs_reduced = skimage.transform.downscale_local_mean(trajs, (16,16,1))

		with env.begin(write = True) as txn:
             		str_id = "{:08}".format(cnt)
             		txn.put(str_id.encode('ascii'), im.astype(np.uint8).dumps());

		with env.begin(write = True) as txn:
             		str_id = "{:08}".format(cnt+1)
             		txn.put(str_id.encode('ascii'), trajs_reduced.astype(np.float32).dumps());
                cnt = cnt + 2;
		imindex = imindex + 1;
        env.close();

def paintArrow(x,y, draw, angle, thecolor):

	angle = angle - math.pi/2.0
	first = math.cos(angle)*0 - math.sin(angle)*(5);
	second = math.sin(angle)*0 + math.cos(angle)*(5);
	top = (x+first,y+second);

	first = math.cos(angle)*(2.5) - math.sin(angle)*0;
	second = math.sin(angle)*(2.5) + math.cos(angle)*0;
	right = (x+first,y+second);

	first = math.cos(angle)*(-2.5) - math.sin(angle)*0;
	second = math.sin(angle)*(-2.5) + math.cos(angle)*0;
	left = (x+first,y+second);

        thelist = [top, right,left, top];
	draw.polygon(thelist,fill=thecolor)

def drawArrows(im,dcttraj,sampledir, curstep, isFine=False):
	fulltraj = np.zeros([32,60,64,80], np.float32);
      
        if(isFine):
                fulltraj[:,0,:,:] = dcttraj[:,0,:,:];
                fulltraj[:,1,:,:] = dcttraj[:,1,:,:];
                fulltraj[:,2,:,:] = dcttraj[:,2,:,:];
                fulltraj[:,3,:,:] = dcttraj[:,3,:,:];
                fulltraj[:,4,:,:] = dcttraj[:,4,:,:];
                fulltraj[:,30,:,:] = dcttraj[:,5,:,:];
                fulltraj[:,31,:,:] = dcttraj[:,6,:,:];
                fulltraj[:,32,:,:] = dcttraj[:,7,:,:];
                fulltraj[:,33,:,:] = dcttraj[:,8,:,:];
                fulltraj[:,34,:,:] = dcttraj[:,9,:,:];
		fulltraj = fulltraj/1000.0;
	else:

		fulltraj[:,0,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,0,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,1,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,1,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,2,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,2,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,3,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,3,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,4,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,4,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,30,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,5,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,31,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,6,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,32,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,7,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,33,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,8,:,:],
				(16,20)), 4.0, order=0)/1000.0)
		fulltraj[:,34,:,:] = (ndimage.interpolation.zoom(np.reshape(dcttraj[:,9,:,:],
				(16,20)), 4.0, order=0)/1000.0)
	width = 80
	height = 64
	res = 2
	xscale = 256/height;
	yscale = 320/width;

	l = range(0,height);
	m = range(0,width);

	ims = [];
	counts = [];
	I = Image.fromarray(np.uint8(im));
	I.save(str(30) + '.jpg')
	for i in range(0,30):
		I = Image.fromarray(np.uint8(im));
		if i == 0:
		  I = I.resize((width,height), Image.ANTIALIAS);
		  I = I.resize((320,256), Image.ANTIALIAS);
		ims.append(np.array(I, np.float64));
		counts.append(np.ones((256,320), np.float64));

	firstI = ims[0];
	secondI = ims[1].copy();
	for i in l:
	  for j in m:
	    x = [j*yscale + .5*yscale];
	    y = [i*xscale + .5*xscale];
	    firstx  = int(x[0] - res*yscale);
	    secondx = int(x[0] + res*yscale);
	    firsty = int(y[0] - res*xscale);
	    secondy = int(y[0] + res*xscale);
	    if firstx < 0:
	      continue
	    if firsty < 0:
	      continue
	    if secondx > 319:
	      continue
	    if secondy > 255:
	      continue
	    pixel = firstI[firsty:secondy,firstx:secondx,:];
	    fulltraj[0,0:30,i,j] = fft.idct(fulltraj[0,0:30,i,j], norm='ortho')
	    fulltraj[0,30:60,i,j] = fft.idct(fulltraj[0,30:60,i,j], norm='ortho')

	for i in range(0,30):
		ims[i][:,:,0] = np.divide(ims[i][:,:,0], counts[i]);
		ims[i][:,:,1] = np.divide(ims[i][:,:,1], counts[i]);
		ims[i][:,:,2] = np.divide(ims[i][:,:,2], counts[i]);

        draws = [];
        finalims = [];
	for i in range(0,30):
		I = Image.fromarray(np.uint8(ims[i]*1.00 + secondI*0.));
		finalims.append(I);
		draw = ImageDraw.Draw(I);
		draws.append(draw);

	l = range(0,height);
	l = l[::4];
	m = range(0,width);
	m = m[::4];

	for i in l:
	  for j in m:
	    x = [j*yscale + .5*yscale];
	    y = [i*xscale + .5*xscale];
	    firstx  = int(x[0] - res*yscale);
	    secondx = int(x[0] + res*yscale);
	    firsty = int(y[0] - res*xscale);
	    secondy = int(y[0] + res*xscale);
	    if firstx < 0:
	      continue
	    if firsty < 0:
	      continue
	    if secondx > 319:
	      continue
	    if secondy > 255:
	      continue
	    if np.mean(abs(fulltraj[0,:,i,j])) < 0.01:
	     continue
	    for k in range(0,30):
		x.append(j*xscale + fulltraj[0,k,i,j]*320 + .5*xscale)
	    for k in range(30,60):
		y.append(i*yscale + fulltraj[0,k,i,j]*256 + .5*yscale)
	    for g in range(1,30):
		firstx  = int(round(x[g]) - res*round(yscale));
		secondx = int(round(x[g]) + res*round(yscale));
		firsty = int(round(y[g]) - res*round(xscale));
		secondy = int(round(y[g]) + res*round(xscale));
		if firstx < 0:
		  continue
		if firsty < 0:
		  continue
		if secondx > 319:
		  continue
		if secondy > 255:
		  continue
		for h in range(max(0,g-30),g):
			xdiff = x[h+1] - x[h];
			ydiff = y[h+1] - y[h];
			xdiff = xdiff/math.sqrt(xdiff*xdiff + ydiff*ydiff)
			ydiff = ydiff/math.sqrt(xdiff*xdiff + ydiff*ydiff)
			angle = math.atan2(ydiff,xdiff);
			angle = (angle + 2*math.pi)/(2*math.pi);
			(r,gr,b) = colorsys.hsv_to_rgb(angle,1.0,1.0);
			thecolor =  (int(r*255.0),int(gr*255.0),int(b*255.0))
			draws[g].line((x[h],y[h],x[h+1],y[h+1]), width=2, fill=thecolor)

		xdiff = x[g] - x[g-1];
		ydiff = y[g] - y[g-1];
		xdiff = xdiff/math.sqrt(xdiff*xdiff + ydiff*ydiff)
		ydiff = ydiff/math.sqrt(xdiff*xdiff + ydiff*ydiff)
		angle = math.atan2(ydiff,xdiff);
		angle = math.atan2(ydiff,xdiff);
		angle = (angle + 2*math.pi)/(2*math.pi);
		(r,gr,b) = colorsys.hsv_to_rgb(angle,1.0,1.0);
		thecolor =  (int(r*255.0),int(gr*255.0),int(b*255.0))
		paintArrow(x[g],y[g], draws[g], ((angle*2*math.pi) - 2*math.pi), thecolor)


	theFrames = [];
	for i in range(0,30):
		theFrames.append(finalims[i]);

	writeGif(sampledir + "/thefig" + str(curstep) + ".gif", theFrames, duration=.07);


def drawPixels(im,dcttraj,sampledir, curstep, isFine=False):

	if(isFine):
                fulltraj = np.zeros([1,60,64,80], np.float32);
                fulltraj[:,0,:,:] = (dcttraj[:,0,:,:]/1000).copy();
                fulltraj[:,1,:,:] = (dcttraj[:,1,:,:]/1000).copy();
                fulltraj[:,2,:,:] = (dcttraj[:,2,:,:]/1000).copy();
                fulltraj[:,3,:,:] = (dcttraj[:,3,:,:]/1000).copy();
                fulltraj[:,4,:,:] = (dcttraj[:,4,:,:]/1000).copy();
                fulltraj[:,30,:,:] = (dcttraj[:,5,:,:]/1000).copy();
                fulltraj[:,31,:,:] = (dcttraj[:,6,:,:]/1000).copy();
                fulltraj[:,32,:,:] = (dcttraj[:,7,:,:]/1000).copy();
                fulltraj[:,33,:,:] = (dcttraj[:,8,:,:]/1000).copy();
                fulltraj[:,34,:,:] = (dcttraj[:,9,:,:]/1000).copy();

                width = 80
                height = 64

	else:
		fulltraj = np.zeros([1,60,16,20], np.float32);
		fulltraj[:,0,:,:] = (dcttraj[:,0,:,:]/1000).copy();
		fulltraj[:,1,:,:] = (dcttraj[:,1,:,:]/1000).copy();
		fulltraj[:,2,:,:] = (dcttraj[:,2,:,:]/1000).copy();
		fulltraj[:,3,:,:] = (dcttraj[:,3,:,:]/1000).copy();
		fulltraj[:,4,:,:] = (dcttraj[:,4,:,:]/1000).copy();
		fulltraj[:,30,:,:] = (dcttraj[:,5,:,:]/1000).copy();
		fulltraj[:,31,:,:] = (dcttraj[:,6,:,:]/1000).copy();
		fulltraj[:,32,:,:] = (dcttraj[:,7,:,:]/1000).copy();
		fulltraj[:,33,:,:] = (dcttraj[:,8,:,:]/1000).copy();
		fulltraj[:,34,:,:] = (dcttraj[:,9,:,:]/1000).copy();

		width = 20
		height = 16
	
        res = 1
	xscale = 256/height;
	yscale = 320/width;

	l = range(0,height);
	m = range(0,width);

	ims = [];
	counts = [];
	I = Image.fromarray(np.uint8(im));
	for i in range(0,30):
		I = Image.fromarray(np.uint8(im));
		if i == 0:
		  I = I.resize((width,height), Image.ANTIALIAS);
		  I = I.resize((320,256), Image.ANTIALIAS);
		ims.append(np.array(I, np.float64));
		counts.append(np.ones((256,320), np.float64));

	firstI = ims[0];
	secondI = ims[1].copy();
	for i in l:
	  for j in m:
	    x = [j*yscale + .5*yscale];
	    y = [i*xscale + .5*xscale];
	    firstx  = int(x[0] - res*yscale);
	    secondx = int(x[0] + res*yscale);
	    firsty = int(y[0] - res*xscale);
	    secondy = int(y[0] + res*xscale);
	    if firstx < 0:
	      continue
	    if firsty < 0:
	      continue
	    if secondx > 319:
	      continue
	    if secondy > 255:
	      continue
	    pixel = firstI[firsty:secondy,firstx:secondx,:];
	    fulltraj[0,0:30,i,j] = fft.idct(fulltraj[0,0:30,i,j], norm='ortho')
	    fulltraj[0,30:60,i,j] = fft.idct(fulltraj[0,30:60,i,j], norm='ortho')
	    if np.mean(abs(fulltraj[0,:,i,j])) < 0.01:
	     continue
	    for k in range(0,30):
		x.append(j*xscale + fulltraj[0,k,i,j]*320 + .5*xscale)
	    for k in range(30,60):
		y.append(i*yscale + fulltraj[0,k,i,j]*256 + .5*yscale)
	    for g in range(1,30):
		firstx  = int(round(x[g]) - res*round(yscale));
		secondx = int(round(x[g]) + res*round(yscale));
		firsty = int(round(y[g]) - res*round(xscale));
		secondy = int(round(y[g]) + res*round(xscale));
		if firstx < 0:
		  continue
		if firsty < 0:
		  continue
		if secondx > 319:
		  continue
		if secondy > 255:
		  continue
		
		if(isFine):
			ims[g][firsty:secondy,firstx:secondx,0] = ims[g][firsty:secondy,firstx:secondx,0] + pixel[:,:,0];
			ims[g][firsty:secondy,firstx:secondx,1] = ims[g][firsty:secondy,firstx:secondx,1] + pixel[:,:,1];
			ims[g][firsty:secondy,firstx:secondx,2] = ims[g][firsty:secondy,firstx:secondx,2] + pixel[:,:,2];
			counts[g][firsty:secondy,firstx:secondx] = counts[g][firsty:secondy,firstx:secondx] + 1;

		else:
			ims[g][firsty:secondy,firstx:secondx,0] = pixel[:,:,0];
			ims[g][firsty:secondy,firstx:secondx,1] = pixel[:,:,1];
			ims[g][firsty:secondy,firstx:secondx,2] = pixel[:,:,2];




	theFrames = [];
	for i in range(0,30):		
		if(isFine):
			ims[i][:,:,0] = np.divide(ims[i][:,:,0], counts[i]);
			ims[i][:,:,1] = np.divide(ims[i][:,:,1], counts[i]);
			ims[i][:,:,2] = np.divide(ims[i][:,:,2], counts[i]);


		I = Image.fromarray(np.uint8(ims[i]*1.00 + secondI*0.));
		theFrames.append(np.uint8(ims[i]*1.00 + secondI*0.));

	writeGif(sampledir + "/thefig_pixel_" + str(curstep) + ".gif", theFrames, duration=.07);
