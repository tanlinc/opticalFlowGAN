#!/usr/bin/python
#execfile('myaddpath.py');
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
from multiprocessing.sharedctypes import Array as sharedArray
import time;
import ctypes
from mako.template import Template
from mako import exceptions
import traceback

def prep_image(im,origsz):
  im=im.astype(np.float32);
  mean=[123,117,104];
  im=im[:,:,[2,1,0]]
  im2=np.zeros((im.shape[0],im.shape[1],im.shape[2]));
  for i in range(0,3):
      im2[:,:,i]=im[:,:,i]-mean[i]
  im2=im2
  return im2;

# A process that loads batches in the background and prepares them
# while the main thread runs stuff on the GPU.
def imgloader(labelq, dataq, batch_sz, pc_id, sharedmem): 
 try:
  env = lmdb.open('testDB', readonly=True);
  miniidx=0;
  sendctr=0;
  random.seed(pc_id);
  np.random.seed(np.int64(pc_id));
  curmem=0;
  mylock=sharedmem[curmem].get_lock();
  while True:
    label=np.zeros([batch_sz,10,64,80],np.float32);
    data=np.zeros([batch_sz,3,im_sz[0],im_sz[1]],np.float32);
    temp = np.zeros([64,80,10],np.float32);
    j=0;
    while(j < batch_sz):
      flip = random.randint(1,10)%2;
      start = time.time();
      im = [];
      if (miniidx > 800):
	        miniidx = 0;

      with env.begin() as txn:
             	str_id = "{:08}".format(miniidx)
	        raw_datum = txn.get(str(str_id));
             	im = np.loads(raw_datum);

      with env.begin() as txn:
             	str_id = "{:08}".format((miniidx+1))
	        raw_datum = txn.get(str_id);
             	temp = np.loads(raw_datum);

      start = time.time();

      if flip:
	im = im[:,::-1,:];
	temp = temp[:,::-1,:];
        temp[:,:,0:5] *= -1.0;
      data[j,0:3:,:,:]=prep_image(im,im_sz).transpose(2,0,1);
      label[j,:,:,:] = (temp.transpose(2,0,1)).copy()*1000;
      j = j + 1
      miniidx = miniidx + 2
    buf=np.frombuffer(sharedmem[curmem].get_obj(), dtype=np.float32).reshape([batch_sz,3,im_sz[0],im_sz[1]]);
    buf[:,:,:,:]=data;
    dataq.put((curmem,label), timeout=6000);

    del mylock;
    curmem=(curmem+1) % len(sharedmem);
    mylock = sharedmem[curmem].get_lock();

 except Exception as e:
  tup2="".join(traceback.format_exception(*sys.exc_info()));
  dataq.put(tup2);
  raise

if 'solver' not in locals():
  outdir = ut.mfilename() + '_out/';
  if not os.path.exists(outdir):
    os.mkdir(outdir);
  else:
    print('====================================================================');
    print('WARNING: opening output log and OVERWRITING EXISTING DATA.  Proceed?');
    raw_input('====================================================================');

  # Log to a file and print to the command line.  Tends to screw up the terminal, but it's
  # worth it.
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
  tee = subprocess.Popen(["tee", outdir + "/out.log"], stdin=subprocess.PIPE)
  os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
  os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

  if os.path.isfile(ut.mfilename() + '_pause'):
    os.remove(ut.mfilename() + '_pause')

  tpl=Template(filename='./solver.prototxt');
  with open(outdir + 'solver.prototxt',"w") as f:
    try:
      f.write(tpl.render(outdir=outdir,base_lr=1e-4,momentum=0.9,weight_decay=0e-4));
    except:
      print(exceptions.text_error_template().render())
      raise
 
  batch_sz=16;
  im_sz = (256,320);
  patch_sz=(256,320);
  tpl_name='./train_fine.prototxt'
  tpl=Template(filename=tpl_name);#f.read());
  django_ctx={"batch_sz":batch_sz,"kl_loss_wt":.01,"pat_sz":patch_sz[0]};
  tosave={};
  with open(outdir + 'network.prototxt',"w") as f:
    try:
      f.write(tpl.render(**django_ctx));
    except:
      print(exceptions.text_error_template().render())
      raise
  
  caffe.set_mode_gpu();
  caffe.set_device(0);
  solver = caffe.get_solver(outdir + '/solver.prototxt');
  solver.net.copy_from('./final_fine.caffemodel')

  # start the background image loading thread
  labelq=Queue(10)
  dataq=[];
  curstep = 0;
  procs=[]
  sharedmem=[]
  idata = np.zeros([batch_sz,13,im_sz[0],im_sz[1]],np.float32);
  ilabel = np.zeros([batch_sz,1,1,1],np.float32)
  for h in range(0,1):
    dataq.append(Queue(4))
    sharedmem.append([sharedArray(ctypes.c_float, batch_sz*3*im_sz[0]*im_sz[1]) for x in range(0,5)])
    procs.append(Process(target=imgloader,args=(labelq,dataq[-1],batch_sz,h, sharedmem[-1])));
    procs[-1].start()
for roundid in range(0,10000000):
  start=time.time();
  (curmem,label)=dataq[curstep % 1].get(timeout=6000)
  print time.time() - start, " queueloaad"
  print dataq[0].qsize()

  with sharedmem[curstep % 1][curmem].get_lock():
  	start=time.time();
        buf=np.frombuffer(sharedmem[curstep % 1][curmem].get_obj(), dtype=np.float32).reshape(batch_sz,3,im_sz[0],im_sz[1]);
	im = (buf[:,0:3,:,:]).copy();	
        traj = label.copy();
  	solver.net.blobs["rawtrajdata"].data[:] = traj[...];

        solver.net.blobs['imdata'].data[:]=(im[...]);
  	print time.time() - start, " dataloaad"
  	start=time.time();
  	solver.step(1)
  	print time.time() - start, " gpu"
        curstep += 1;
  	dobreak=False;
  	broken=[];
     
  	if curstep % 50 == 0:
    	  themag = solver.net.blobs["trajnorm"].data;
    	  thepredmag = solver.net.blobs["conv2reconst"].data;
    	  thepred = solver.net.blobs["conv1reconst"].data;
    	  thedata = solver.net.blobs["normtrajdata"].data;
    	  theim = solver.net.blobs["imdata"].data;
    	  thezs = solver.net.blobs["thezsample"].data;
          np.save(outdir + "/themag.npy", themag);
          np.save(outdir + "/thepredmag.npy", thepredmag);
          np.save(outdir + "/thepred.npy", thepred);
          np.save(outdir + "/thedata.npy", thedata);
          np.save(outdir + "/thez.npy", thezs);
          np.save(outdir + "/theim.npy", theim);

	  thepred = np.multiply(thepred,thepredmag);
	  thedata = np.multiply(thedata,themag);

	  theim = theim.transpose(0,2,3,1);

	  mean=[123,117,104];
	  I2=np.zeros((theim.shape[0],theim.shape[1],theim.shape[2],theim.shape[3]));
	  for i in range(0,3):
	    I2[:,:,:,i]=theim[:,:,:,i]+np.mean(mean[i]);

	  tempone = I2[:,:,:,0].copy()
	  temptwo = I2[:,:,:,1].copy()
	  tempthree = I2[:,:,:,2].copy()
	  I2[:,:,:,0] = tempthree
	  I2[:,:,:,1] = temptwo
	  I2[:,:,:,2] = tempone

          for hhh in range(0,batch_sz):	
	  	ut.drawArrows(I2[hhh,:,:,:],np.reshape(thepred[hhh,:,:,:], (1,10,16,20)),outdir, hhh);
	  	ut.drawPixels(I2[hhh,:,:,:],np.reshape(thepred[hhh,:,:,:], (1,10,16,20)),outdir, hhh);

	  	ut.drawArrows(I2[hhh,:,:,:],np.reshape(thedata[hhh,:,:,:], (1,10,16,20)),outdir, hhh+16);
	  	ut.drawPixels(I2[hhh,:,:,:],np.reshape(thedata[hhh,:,:,:], (1,10,16,20)),outdir, hhh+16);

  	if os.path.isfile(ut.mfilename() + '_pause'):
          break;
