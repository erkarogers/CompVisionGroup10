import sys
import pickle
import cv2
import fullAlg
import time
import cProfile
import re
import pstats
from pstats import SortKey


def main():
    num_batch_files = 1
    num_images = 0
    images = {}
    image_index = 0
    image_hashes_paper = {}
    time0 = time.time()
    for image_batch in range(num_batch_files):
        print('../Dataset/image_batch_' + str(image_batch) + '.pkl')
        with open('../Dataset/image_batch_' + str(image_batch) + '.pkl' , 'rb') as f:
            images = pickle.load(f)
        num_images += len(images)
        for image_id in images:
            # print(images[image_id].shape)
            image_hashes_paper[image_id] = fullAlg.applyAlg(images[image_id])
            print("Image " + str(image_index) + "/" + str(num_images))
            image_index += 1
            # if image_index > 20:
            #     break
    time1 = time.time()
    dtime = time1 - time0
    print("Total time: " + str(dtime) + "Time/image: " + str(dtime/num_images))

if __name__ == "__main__":
    cProfile.run('main()', "benchmark_profile")
    p = pstats.Stats('benchmark_profile')
    # p.strip_dirs().sort_stats(-1).print_stats()
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
    # main()