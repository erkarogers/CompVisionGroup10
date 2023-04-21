import filtering
import grabcut
import mergeAverage
import backCut

import cv2
from skimage import img_as_ubyte
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops, find_contours
from skimage.color import rgb2gray, label2rgb
import matplotlib.pyplot as plt

def applyAlg(img):
    # cv2.imshow(('img'),img)

    
    filtered_img = filtering.applyFilters(img)
    # cv2.imwrite("./test/imag/tigerWoods_filtered.jpg", filtered_img)

    # print("filtered_img: " + str(filtered_img.shape))
    # cv2.imshow(('filtered_img'),filtered_img)

    
    grabcut_img = grabcut.grabcut(img)
    # grabcut_img = backCut.cut(img)
    # print("grabcut_img: " + str(grabcut_img.shape))
    # cv2.imshow(('grabcut_img'),grabcut_img)
    # cv2.waitKey(0)
    
    grey_grabcut = rgb2gray(grabcut_img)
    
    labeled_grabcut = label(grey_grabcut > 0)
    
    # image_label_overlay = label2rgb(labeled_grabcut, image=grabcut_img, bg_label=0)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    
    cut_props = regionprops(labeled_grabcut, grabcut_img)
    # print("props: " + str(cut_props))

    
    largest_area = 0
    largest_bbox = None
    for prop in cut_props:
        if prop.area > largest_area:
            largest_area = prop.area
            largest_bbox = prop.bbox
            
    # print(largest_area, " : ", largest_bbox)
    if largest_bbox is None:
        cropped_filtered_img = filtered_img
        cropped_grabcut_img = grabcut_img
    else:
        cropped_filtered_img = filtered_img[largest_bbox[0]:largest_bbox[2], largest_bbox[1]:largest_bbox[3]]
        cropped_grabcut_img = grabcut_img[largest_bbox[0]:largest_bbox[2], largest_bbox[1]:largest_bbox[3]]
    # cv2.imwrite("./test/imag/tigerWoods_grabcut.jpg", grabcut_img)

    
    merged_img = mergeAverage.merge_imag(cropped_filtered_img, cropped_grabcut_img, 128)
    
    greyscale_merged_img = img_as_ubyte(rgb2gray(merged_img))
    # cv2.imwrite("./test/imag/tigerWoods_merged.jpg", greyscale_merged_img)
    
    hashed_img = mergeAverage.average_imag(greyscale_merged_img)
    # print("Hashed Image: ")
    # print(hashed_img)
    return hashed_img, largest_bbox is None

def applyAlgImproved(img):
    # cv2.imshow(('img'),img)

    
    filtered_img = filtering.applyFilters(img)
    cv2.imwrite("./test/imag/tigerWoods_filtered.jpg", filtered_img)

    
    # print("filtered_img: " + str(filtered_img.shape))
    # cv2.imshow(('filtered_img'),filtered_img)

    
    # grabcut_img = grabcut.grabcut(img)
    grabcut_img = backCut.cut(img)
    # print("grabcut_img: " + str(grabcut_img.shape))
    # cv2.imshow(('grabcut_img'),grabcut_img)
    cv2.waitKey(0)
    
    
    grey_grabcut = rgb2gray(grabcut_img)
    
    labeled_grabcut = label(grey_grabcut > 0)
    
    # image_label_overlay = label2rgb(labeled_grabcut, image=grabcut_img, bg_label=0)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
    
    cut_props = regionprops(labeled_grabcut, grabcut_img)
    # print("props: " + str(cut_props))

    
    largest_area = 0
    largest_bbox = None
    for prop in cut_props:
        if prop.area > largest_area:
            largest_area = prop.area
            largest_bbox = prop.bbox
            
    # print(largest_area, " : ", largest_bbox)
    if largest_bbox is None:
        cropped_filtered_img = filtered_img
        cropped_grabcut_img = grabcut_img
    else:
        cropped_filtered_img = filtered_img[largest_bbox[0]:largest_bbox[2], largest_bbox[1]:largest_bbox[3]]
        cropped_grabcut_img = grabcut_img[largest_bbox[0]:largest_bbox[2], largest_bbox[1]:largest_bbox[3]]
    
    
    cv2.imwrite("./test/imag/tigerWoods_grabcut.jpg", grabcut_img)
    
    merged_img = mergeAverage.merge_imag(cropped_filtered_img, cropped_grabcut_img, 128)
    
    greyscale_merged_img = img_as_ubyte(rgb2gray(merged_img))
    cv2.imwrite("./test/imag/tigerWoods_merged.jpg", greyscale_merged_img)
    
    hashed_img = mergeAverage.average_imag(greyscale_merged_img)
    # print("Hashed Image: ")
    # print(hashed_img)
    return hashed_img
    
    
def main():
    jsImag = cv2.imread("./test/imag/tigerWoods.jpg")
    (h,w) = jsImag.shape[:2]
    scale = 512 / w
    dim = (int(w * scale), int(h * scale))
    jsImag_res = cv2.resize(jsImag, dim, interpolation = cv2.INTER_AREA)
    print(applyAlg(jsImag_res))


if __name__ == "__main__":
    main()