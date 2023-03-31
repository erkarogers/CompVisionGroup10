#Grabcut algorithm 
#NOTE: Fill 'INSERT PATH HERE' with image path before running

import numpy
import cv2
from matplotlib import pyplot as plt

#reads image
cutImage = cv2.imread(INSERT PATH HERE)

#creates mask
mask = numpy.zeros(cutImage.shape[:2], numpy.uint8)

#background and foreground models
background = numpy.zeros((1,65), numpy.float64))
foreground = numpy.zeros((1,65), numpy.float64))

#(x, y, width, height)
ROI = (0,0,128,128)

#grabcut
cv2.grabCut(cutImage, mask, ROI, background, foreground, 3, cv2.GC_INIT_WITH_RECT)

#new mask
newMask = numpy.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

#processed image
cutImage = cutImage * newMask[:, :, numpy.newaxis]
