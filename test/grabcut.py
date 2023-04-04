#Grabcut algorithm taken from https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
#NOTE: Fill 'INSERT PATH HERE' with image path before running

import numpy
import cv2
from matplotlib import pyplot as plt

def main():
    #reads image
    img = cv2.imread("./test/imag/jordanSpieth.jpg")
    (h,w) = img.shape[:2]
    scale = 512 / w
    dim = (int(w * scale), int(h * scale))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    
    cut_img = grabcut(img)
    
    cv2.imshow("test", cut_img)
    cv2.waitKey(0)


def grabcut(img):
    (w,h) = img.shape[:2]
    #creates mask
    mask = numpy.zeros(img.shape[:2], numpy.uint8)

    #background and foreground models
    background = numpy.zeros((1,65), numpy.float64)
    foreground = numpy.zeros((1,65), numpy.float64)

    #(x, y, width, height)
    ROI = (1,1,w-1,h-1)

    #grabcut
    cv2.grabCut(img, mask, ROI, background, foreground, 3, cv2.GC_INIT_WITH_RECT)

    #new mask
    newMask = numpy.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

    #processed image
    return img * newMask[:, :, numpy.newaxis]


if __name__ == "__main__":
    main()