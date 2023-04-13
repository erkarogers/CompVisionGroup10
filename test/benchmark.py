import sys
import pickle
import cv2
import time
import cProfile
import re
import pstats
from pstats import SortKey
import imagehash
from PIL import Image

import fullAlg


def main():
    num_batch_files = 148
    num_images = 0
    images = {}
    image_index = 0
    image_hashes_paper = {}
    image_hashes_improved = {}
    image_hashes_average = {}
    image_hashes_phash = {}
    image_hashes_wavelet = {}
    image_hashes_diff = {}
    time0 = time.time()
    for image_batch in range(num_batch_files):
        print('../Dataset/image_batch_' + str(image_batch) + '.pkl')
        with open('../Dataset/image_batch_' + str(image_batch) + '.pkl' , 'rb') as f:
            images = pickle.load(f)
        num_images += len(images)
        for image_id in images:
            # print(images[image_id].shape)
            image = Image.fromarray(images[image_id])
            image_hashes_paper[image_id] = fullAlg.applyAlg(images[image_id])
            image_hashes_improved[image_id] = fullAlg.applyAlgImproved(images[image_id])
            image_hashes_average[image_id] = imagehash.average_hash(image)
            image_hashes_phash[image_id] = imagehash.phash(image)
            image_hashes_wavelet[image_id] = imagehash.whash(image)
            image_hashes_diff[image_id] = imagehash.dhash(image)
            
            print("Image " + str(image_index) + "/" + str(num_images))
            image_index += 1
            if image_index % 1000 == 0:
                hashes = (image_hashes_paper, image_hashes_improved, image_hashes_average, image_hashes_phash, image_hashes_wavelet, image_hashes_diff)
                with open('../Dataset/image_hashes' + str(int(image_index/1000)) + '.pkl' , 'wb') as f:
                    pickle.dump(hashes, f)
                image_hashes_paper = {}
                image_hashes_improved = {}
                image_hashes_average = {}
                image_hashes_phash = {}
                image_hashes_wavelet = {}
                image_hashes_diff = {}
    hashes = (image_hashes_paper, image_hashes_improved, image_hashes_average, image_hashes_phash, image_hashes_wavelet, image_hashes_diff)
    with open('../Dataset/image_hashes' + str(int(image_index/1000)) + '.pkl' , 'wb') as f:
        pickle.dump(hashes, f)
    time1 = time.time()
    dtime = time1 - time0
    print("Total time: " + str(dtime) + "Time/image: " + str(dtime/num_images))

if __name__ == "__main__":
    cProfile.run('main()', "benchmark_profile")
    p = pstats.Stats('benchmark_profile')
    # p.strip_dirs().sort_stats(-1).print_stats()
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
    # main()