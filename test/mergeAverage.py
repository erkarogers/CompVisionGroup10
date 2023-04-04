# To run test:
# from test import *
# a = test()

import cv2
import numpy as np

'''
Merges two images and outputs them in the given size.

@param img1 First image to be merged
@param img2 Second image to be merged
@param size (nxn) Size of image to be output

@return the merged (size x size) image (mat)
'''
def merge_imag(img1, img2, size):
  dim = (int(size),int(size))
  img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
  img2 = cv2.resize(img2, (int(size),int(size)))
  return cv2.add(img1, img2)

'''
Averages pixel using algorithm based in A Revised Averaging Algorithm for an Effective Feature Extraction in Component- based Image Retrieval System

@param img Image holding the pixel we want to average
@param x X coordinate of pixel to be averaged
@param y Y coordinate of pixel to be averaged
@param n Size of matrix the pixel belongs to

@returns Averaged pixel based upon afformentioned algorithm
'''
def average_pixel(img, x, y, n):
  sum = 0
  for i in range(n):
    for j in range(n):
      sum += (((n-i) * (n-j)) / (n*n)) * img[x+i, y+j]
  return sum / (n*n)

'''
Average image for compression using algorithm based in A Revised Averaging Algorithm for an Effective Feature Extraction in Component- based Image Retrieval System

@param img Image to be averaged for compression

@return Compressed image (numpy array)
'''
def average_imag(img):
  size = img.shape[0]
  aSize = size

  # calculate size of array
  z = 0
  while aSize >= 16:
    for i in range(int(size/aSize)):
      for j in range(int(size/aSize)):
        z += 1
    aSize = aSize / 2
  
  # allocate array w/ zeros to increase speed
  a = np.zeros(z)
  n = size
  z = 0;
  while n >= 16 :
    for i in range(int(size/n)):
      for j in range(int(size/n)):
        # print(average_pixel(img, i, j, int(n)))
        a[z] = average_pixel(img, i, j, int(n))
        z += 1
    n = n / 2
  
  return a
        
        
def test():
  jsImag = cv2.imread("./imag/jordanSpieth.jpg")
  twImag = cv2.imread("./imag/tigerWoods.jpg")
  
  mergedImag = merge_imag(jsImag, twImag, 128)
 
  # Uncomment to display merged image
  #  cv2.imshow("Merged Image", mergedImag)
  #  cv2.waitKey(0)
  #  cv2.destroyAllWindows()
  #  cv2.waitKey(1)

  return average_imag(mergedImag)

