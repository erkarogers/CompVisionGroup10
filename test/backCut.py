import cv2
import numpy as np


def cut(img):
    # Parameters
    canny_low = 15
    canny_high = 200
    min_area = 0.0009
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    blur_size = 3
    mask_color = [0.0,0.0,0.0]
    image_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    stretched_yuv = image_yuv
    stretched_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    stretched = cv2.cvtColor(stretched_yuv, cv2.COLOR_YUV2BGR)
    # cv2.imshow("stretched", stretched)

    # Convert image to grayscale        
    image_gray = cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY)
    
    blured = cv2.GaussianBlur(image_gray, (blur_size, blur_size),0)
    # Apply Canny Edge Dection
    edges = cv2.Canny(blured, canny_low, canny_high)
    
    # cv2.imshow("canny", edges)
    
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    
    # get the contours and their areas
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = contours
    # print(contours)
    contour_info = [(c, cv2.contourArea(c),) for c in cnt]
    
    # Get the area of the image as a comparison
    image_area = img.shape[0] * img.shape[1]  
    
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area
    
    
    
    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype = np.uint8)
    max_contour_area = 0
    min_contour_area = 1000000000000
    
    # Go through and find relevant contours and apply to mask
    for contour in contour_info: 
        if contour[1] > max_contour_area:
            max_contour_area = contour[1]
        if contour[1] < min_contour_area:
            min_contour_area = contour[1]
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))

    # print("max: " + str(max_contour_area) + " min: " + str(min_contour_area))
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    
    # Ensures data types match up
    mask_stack = mask.astype('float32') / 255.0
    mask_stack = cv2.cvtColor(mask_stack, cv2.COLOR_GRAY2RGB)
    frame = img.astype('float32') / 255.0
    
    # cv2.imshow("mask", mask_stack)
    

    
    # Blend the image and the mask
    masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')
    # cv2.imshow("masked", masked)
    # cv2.waitKey(0)
    return masked