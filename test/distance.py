import pickle
import cv2
import time
import cProfile
import pstats
from pstats import SortKey
import imagehash
import numpy as np
import binascii
from annoy import AnnoyIndex
import random

def merge(list1, list2):
     
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def getImage(id):
    foundImage = False
    image_batch_index = 0
    image_batch_count = 148
    image = None
    while image_batch_index < image_batch_count and not foundImage:
        image_batch = {}
        with open('../Dataset/image_batch_' + str(image_batch_index) + '.pkl' , 'rb') as f:
            image_batch = pickle.load(f)
        # print("searching ", image_batch_index, "/", image_batch_count-1)
        if id in image_batch.keys():
            return image_batch[id]
        image_batch_index += 1
        
def getCollage(images):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (125, 125))
    col0 = np.vstack([images[0], images[1], images[2]])
    col1 = np.vstack([images[3], images[4], images[5]])
    col2 = np.vstack([images[6], images[7], images[8]])
    return np.hstack([col0, col1, col2])


# a lot of this was based of of the blog post https://lvngd.com/blog/image-similarity-nearest-neighbors/ 

def main():
    num_batch_files = 5
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
        print('./Data/image_hashes' + str(image_batch+1) + '.pkl')
        image_hashes_paper_tmp = {}
        image_hashes_improved_tmp = {}
        image_hashes_average_tmp = {}
        image_hashes_phash_tmp = {}
        image_hashes_wavelet_tmp = {}
        image_hashes_diff_tmp = {}
        with open('./Data/image_hashes' + str(image_batch+1) + '.pkl' , 'rb') as f:
            [image_hashes_paper_tmp, image_hashes_improved_tmp, image_hashes_average_tmp, image_hashes_phash_tmp, image_hashes_wavelet_tmp, image_hashes_diff_tmp] = pickle.load(f)
            # image_hashes_paper_tmp = pickle.load(f)
            # image_hashes_improved_tmp = pickle.load(f)
            # image_hashes_average_tmp = pickle.load(f)
            # image_hashes_phash_tmp = pickle.load(f)
            # image_hashes_wavelet_tmp = pickle.load(f)
            # image_hashes_diff_tmp = pickle.load(f)
        num_images += len(image_hashes_paper)
        image_hashes_paper.update(image_hashes_paper_tmp)
        image_hashes_improved.update(image_hashes_improved_tmp)
        image_hashes_average.update(image_hashes_average_tmp)
        image_hashes_phash.update(image_hashes_phash_tmp)
        image_hashes_wavelet.update(image_hashes_wavelet_tmp)
        image_hashes_diff.update(image_hashes_diff_tmp)
    
    hash_2_id = {}
    for key, value in image_hashes_paper.items():
        hashed = hash(key) % 100000
        hash_2_id[hashed] = key
    
    paper_hashes = []
    paper_hashes_image_hash = {}
    paper_length = 0
    for id in image_hashes_paper:
        int_id = hash(id) % 100000
        # binary = str(binascii.b2a_hex(image_hashes_paper[id].round().astype(np.uint8).tobytes()))[2:-1]
        # hashed = imagehash.hex_to_flathash(binary, 170)
        # paper_hashes_image_hash[int_id] = hashed.hash.astype('int').flatten()
        paper_hashes_image_hash[int_id] = image_hashes_paper[id]
        paper_length = paper_hashes_image_hash[int_id].shape[0]
        
    paper_annoy = AnnoyIndex(paper_length, "euclidean")
    for key, value in paper_hashes_image_hash.items():
        paper_annoy.add_item(key, value)
    
    paper_annoy.build(500, -1)
    
    test_query_id, value = random.choice(list(image_hashes_paper.items()))
    test_query_id_hash = hash(test_query_id) % 100000
    paper_neighbors = paper_annoy.get_nns_by_item(test_query_id_hash, 9, include_distances=True)
    print("ID: ", test_query_id_hash, "Paper_neighbors: ", paper_neighbors)
    images = []
    for image_id_hash, dist in merge(paper_neighbors[0], paper_neighbors[1]):
        id = hash_2_id[image_id_hash]
        images.append(getImage(id))
    collage = getCollage(images)
    cv2.imwrite("./Data/paper_collage.jpg", collage)
    cv2.imshow("Paper Collage", collage)


    # cv2.waitKey(0)
    
    paper_hashes_improved = []
    paper_hashes_improved_image_hash = {}
    paper_improved_length = 0
    for id in image_hashes_improved:
        int_id = hash(id) % 100000
        # binary = str(binascii.b2a_hex(image_hashes_improved[id].round().astype(np.uint8).tobytes()))[2:-1]
        # hashed = imagehash.hex_to_flathash(binary, 170)
        # paper_hashes_improved_image_hash[int_id] = hashed.hash.astype('int').flatten()
        paper_hashes_improved_image_hash[int_id] = image_hashes_improved[id]
        paper_improved_length = paper_hashes_improved_image_hash[int_id].shape[0]
        
    paper_improved_annoy = AnnoyIndex(paper_improved_length, "euclidean")
    for key, value in paper_hashes_improved_image_hash.items():
        paper_improved_annoy.add_item(key, value)
        
    paper_improved_annoy.build(500, -1)
    
    paper_improved_neighbors = paper_improved_annoy.get_nns_by_item(test_query_id_hash, 9, include_distances=True)
    print("ID: ", test_query_id_hash, "Paper_improved_neighbors: ", paper_improved_neighbors)
    images = []
    for image_id_hash, dist in merge(paper_improved_neighbors[0], paper_improved_neighbors[1]):
        id = hash_2_id[image_id_hash]
        images.append(getImage(id))
    collage = getCollage(images)
    cv2.imwrite("./Data/paper_improved_collage.jpg", collage)
    cv2.imshow("Paper Improved Collage", collage)
    
    
    average_hashes = {}
    average_length = 0
    for id in image_hashes_average:
        int_id = hash(id) % 100000
        average_hashes[int_id] = image_hashes_average[id].hash.astype('int').flatten()
        average_length = average_hashes[int_id].shape[0]
    
    average_annoy = AnnoyIndex(average_length, "hamming")
    for key, value in average_hashes.items():
        average_annoy.add_item(key, value)
        
    average_annoy.build(500, -1)

    average_neighbors = average_annoy.get_nns_by_item(test_query_id_hash, 9, include_distances=True)
    print("ID: ", test_query_id_hash, "Average_neighbors: ", average_neighbors)
    images = []
    for image_id_hash, dist in merge(average_neighbors[0], average_neighbors[1]):
        id = hash_2_id[image_id_hash]
        images.append(getImage(id))
    collage = getCollage(images)
    cv2.imwrite("./Data/average_collage.jpg", collage)
    cv2.imshow("Average Collage", collage)
    
    phash_hashes = {}
    phash_length = 0
    for id in image_hashes_phash:
        int_id = hash(id) % 100000
        phash_hashes[int_id] = image_hashes_phash[id].hash.astype('int').flatten()
        phash_length = phash_hashes[int_id].shape[0]
    
    phash_annoy = AnnoyIndex(phash_length, "hamming")
    for key, value in phash_hashes.items():
        phash_annoy.add_item(key, value)
        
    phash_annoy.build(500, -1)

    phash_neighbors = phash_annoy.get_nns_by_item(test_query_id_hash, 9, include_distances=True)
    print("ID: ", test_query_id_hash, "phash_neighbors: ", phash_neighbors)
    images = []
    for image_id_hash, dist in merge(phash_neighbors[0], phash_neighbors[1]):
        id = hash_2_id[image_id_hash]
        images.append(getImage(id))
    collage = getCollage(images)
    cv2.imwrite("./Data/phash_collage.jpg", collage)
    cv2.imshow("phash Collage", collage)
    
    wavelet_hashes = {}
    wavelet_length = 0
    for id in image_hashes_wavelet:
        int_id = hash(id) % 100000
        wavelet_hashes[int_id] = image_hashes_wavelet[id].hash.astype('int').flatten()
        wavelet_length = wavelet_hashes[int_id].shape[0]
    
    wavelet_annoy = AnnoyIndex(wavelet_length, "hamming")
    for key, value in wavelet_hashes.items():
        wavelet_annoy.add_item(key, value)
        
    wavelet_annoy.build(500, -1)

    wavelet_neighbors = wavelet_annoy.get_nns_by_item(test_query_id_hash, 9, include_distances=True)
    print("ID: ", test_query_id_hash, "wavelet_neighbors: ", wavelet_neighbors)
    images = []
    for image_id_hash, dist in merge(wavelet_neighbors[0], wavelet_neighbors[1]):
        id = hash_2_id[image_id_hash]
        images.append(getImage(id))
    collage = getCollage(images)
    cv2.imwrite("./Data/wavelet_collage.jpg", collage)
    cv2.imshow("wavelet Collage", collage)
    
    diff_hashes = {}
    diff_length = 0
    for id in image_hashes_diff:
        int_id = hash(id) % 100000
        diff_hashes[int_id] = image_hashes_diff[id].hash.astype('int').flatten()
        diff_length = diff_hashes[int_id].shape[0]
    
    diff_annoy = AnnoyIndex(diff_length, "hamming")
    for key, value in diff_hashes.items():
        diff_annoy.add_item(key, value)
        
    diff_annoy.build(500, -1)

    diff_neighbors = diff_annoy.get_nns_by_item(test_query_id_hash, 9, include_distances=True)
    print("ID: ", test_query_id_hash, "diff_neighbors: ", diff_neighbors)
    images = []
    for image_id_hash, dist in merge(diff_neighbors[0], diff_neighbors[1]):
        id = hash_2_id[image_id_hash]
        images.append(getImage(id))
    collage = getCollage(images)
    cv2.imwrite("./Data/diff_collage.jpg", collage)
    cv2.imshow("diff Collage", collage)
    
    cv2.waitKey(0)



if __name__ == "__main__":
    cProfile.run('main()', "benchmark_profile")
    p = pstats.Stats('benchmark_profile')
    # p.strip_dirs().sort_stats(-1).print_stats()
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
    # main()