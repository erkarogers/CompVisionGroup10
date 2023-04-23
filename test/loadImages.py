import http
import numpy as np
import pandas as pd
import glob
import urllib.request
import cv2
import sys
import pickle


def url_to_image(url, readFlag=cv2.IMREAD_COLOR, width=1024):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    # print(url)
    try:
      resp = urllib.request.urlopen(url + "?width=" + str(width))
    except urllib.error.HTTPError:
      return None
    except http.client.InvalidURL:
      return None
    except urllib.error.URLError:
      return None
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # print(len(image))
    image = cv2.imdecode(image, readFlag)
    # return the image
    return image

path = '../Dataset/'
documents = ['photos', 'keywords', 'collections', 'conversions', 'colors']
datasets = {}

for doc in documents:
  files = glob.glob(path + doc + ".tsv*")

  subsets = []
  for filename in files:
    df = pd.read_csv(filename, sep='\t', header=0)
    subsets.append(df)

  datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

# print(datasets['photos'].head())
print("Done Reading database files")
image_batch = {}
batch_size = 0
image_batch_index = 0
image_index = 0
num_rows = len(datasets['photos'].index)

for index, image in datasets['photos'].iterrows():
  # print(image)
  print("Image #" + str(index) + "/" + str(num_rows))
  img = url_to_image(image['photo_image_url'], width=512)
  if not(img is None):
    size = sys.getsizeof(img)
    batch_size += size
    # print(batch_size)
    # cv2.imshow('Test', img)
    # cv2.waitKey(0)
    image_batch[str(image['photo_id'])] = img
    if batch_size > 1024*1024*128:
      print("Batch #" + str(image_batch_index) + "\tImage: " + str(index) + "/" + str(num_rows))
      # print(sys.getsizeof(image_batch))
      with open('../Dataset/image_batch_' + str(image_batch_index) + '.pkl' , 'wb') as f:
        pickle.dump(image_batch, f)
      image_batch = {}
      batch_size = 0
      image_batch_index += 1

