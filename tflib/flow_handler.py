import numpy as np
import os
import sys
import cv2

#   compute colored image to visualize optical flow file .flo

#   According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
#   Contact: dqsun@cs.brown.edu
#   Contact: schar@middlebury.edu

# from: https://github.com/Johswald/flow-code-python
#   Author: Johannes Oswald, Technical University Munich
#   Contact: johannes.oswald@tum.de
#   Date: 26/04/2017

#	For more information, check http://vision.middlebury.edu/flow/ 

TAG_STRING = 'PIEH'
TAG_FLOAT = 202021.25

def read_flo_file(filename):
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert os.path.isfile(filename) is True, "file does not exist %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % filename[-4:]
    f = open(filename,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0] # need [0]!
    assert flo_number == TAG_FLOAT, 'Magic Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0] # need 0!
    h = np.fromfile(f, np.int32, count=1)[0] # need 0!
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()
    return flow


def write_flo_file(flow, filename):
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]

    height, width, nBands = flow.shape
    assert nBands == 2, "Number of bands = %r != 2" % nBands
    u = flow[: , : , 0]
    v = flow[: , : , 1]	
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    f = open(filename,'wb')
    f.write(TAG_STRING)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()

def makeColorwheel():
	# color encoding scheme, adapted from the color circle idea described at
	# http://members.shaw.ca/quadibloc/other/colint.htm

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR
	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255;
	col += YG;

	#GC
	colorwheel[col:GC+col, 1]= 255 
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC;

	#CB
	colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB;

	#BM
	colorwheel[col:BM+col, 2]= 255 
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM;

	#MR
	colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return 	colorwheel

def computeColor(u, v):
	colorwheel = makeColorwheel() #;
        # NaN handling
	nan_u = np.where(np.isnan(u))
	nan_v = np.where(np.isnan(v)) 
	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0 
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) /2 * (ncols-1) # -1~1 mapped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1;
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1],3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:,i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
		col[~idx] *= 0.75 # out of range
		img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)


def computeImg(flow):
	eps = sys.float_info.epsilon
	UNKNOWN_FLOW_THRESH = 1e9
	UNKNOWN_FLOW = 1e10

	u = flow[: , : , 0]
	v = flow[: , : , 1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999

	maxrad = -1
	#fix unknown flow
	greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
	greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
	u[greater_u] = 0
	u[greater_v] = 0
	v[greater_u] = 0 
	v[greater_v] = 0

	maxu = max([maxu, np.amax(u)])
	minu = min([minu, np.amin(u)])

	maxv = max([maxv, np.amax(v)])
	minv = min([minv, np.amin(v)])
	rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v)) 
	maxrad = max([maxrad, np.amax(rad)])
	print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

	u = u/(maxrad+eps)
	v = v/(maxrad+eps)
	img = computeColor(u, v)  # gives uint8

	img_inverted = np.zeros_like(img)  # swap red and blue
	img_inverted[:,:,0] = img[:,:,2]
	img_inverted[:,:,1] = img[:,:,1]
	img_inverted[:,:,2] = img[:,:,0]
	return img_inverted

