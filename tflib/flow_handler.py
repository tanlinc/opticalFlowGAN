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

    size = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((3, size))

    col = 0
    # RY
    colorwheel[0, col:col+RY] = 255
    colorwheel[1, col:col+RY] = np.floor(255 * np.arange(RY)/RY)
    col += RY

    # YG
    colorwheel[0, col:col+YG] = 255 - np.floor(255 * np.arange(YG)/YG)
    colorwheel[1, col:col+YG] = 255
    col += YG

    # GC
    colorwheel[1, col:col+GC] = 255
    colorwheel[2, col:col+GC] = np.floor(255 * np.arange(GC)/GC)
    col += GC

    # CB
    colorwheel[1, col:col+CB] = 255 - np.floor(255 * np.arange(CB)/CB)
    colorwheel[2, col:col+CB] = 255
    col += CB

    # BM
    colorwheel[0, col:col+BM] = np.floor(255 * np.arange(BM)/BM)
    colorwheel[2, col:col+BM] = 255
    col += BM

    # MR
    colorwheel[0, col:col+MR] = 255
    colorwheel[2, col:col+MR] = 255 - np.floor(255 * np.arange(MR)/MR)

    return colorwheel.astype('uint8');


def computeNormalizedFlow(u, v, max_flow=-1, min_max_flow = -1):
    
    eps = 1e-15
    UNKNOWN_FLOW_THRES = 1e9
    UNKNOWN_FLOW = 1e10

    maxu = -999
    maxv = -999
    minu = 999
    minv = 999
    maxrad = -1

    # fix unknown flow
    idxUnknown = np.where(np.logical_or(np.abs(u) > UNKNOWN_FLOW_THRES, np.abs(v) > UNKNOWN_FLOW_THRES))
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    #maxu = np.maximum(maxu, np.max(u))
    #minu = np.minimum(minu, np.min(u))
    #maxv = np.maximum(maxv, np.max(v))
    #minv = np.minimum(minv, np.min(v))
    
    if max_flow < 0:
        rad = np.sqrt(u**2 + v**2)
        if min_max_flow >=0:
            rad = np.max((np.max(rad), min_max_flow)) # lower bound for max_flow => don't amplifiy noise
    else:
        rad = max_flow #biggest allowed flow = max_flow
    maxrad = np.max(rad)

    #print("max flow: ", maxrad, " flow range: u = ", minu, "..", maxu, "v = ", minv, "..", maxv)

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    return u, v

def computeFlowImg(flow, max_flow=-1,min_max_flow=-1):  
    u, v = flow[:,:,0], flow[:,:,1]
    u, v = computeNormalizedFlow(u, v, max_flow,min_max_flow)

    nanIdx = np.logical_or(np.isnan(u), np.isnan(v))
    u[np.where(nanIdx)] = 0
    v[np.where(nanIdx)] = 0

    cw = makeColorwheel().T

    M, N = u.shape
    img = np.zeros((M, N, 3)).astype('uint8')

    mag = np.sqrt(u**2 + v**2)
    
    phi = np.arctan2(-v, -u) / np.pi # [-1, 1]
    phi_idx = (phi + 1.0) / 2.0 * (cw.shape[0] - 1)
    f_phi_idx = np.floor(phi_idx).astype('int')

    c_phi_idx = f_phi_idx + 1
    c_phi_idx[c_phi_idx == cw.shape[0]] = 0

    floor = phi_idx - f_phi_idx

    for i in range(cw.shape[1]):
        tmp = cw[:,i]
        
        # linear blend between colors
        col0 = tmp[f_phi_idx] / 255.0 # from colorwheel take specified values in phi_idx
        col1 = tmp[c_phi_idx] / 255.0
        col = (1.0 - floor)*col0 + floor * col1

        # increase saturation for small magnitude
        sat_idx = np.where(mag <= 1)
        non_sat_idx = np.where(mag > 1)
        col[sat_idx] = 1 - mag[sat_idx] * (1 - col[sat_idx])

        col[non_sat_idx] = col[non_sat_idx] * 0.75

        img[:,:, i] = (np.floor(255.0*col*(1-nanIdx))).astype('uint8')
    return img

def computeColor(u, v):  # old 
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


def computeImg(flow):  # old 
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

