import cv2
import numpy as np
import h5py
f = h5py.File('../tmp/best.h5', 'r')
dset = f['key']
data = np.array(dset[:,:,:])
file = '../tmp/best.jpg'
cv2.imwrite(file, data)
