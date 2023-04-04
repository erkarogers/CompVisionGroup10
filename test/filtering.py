import cv2
import numpy as np

'''
Generates a filtered image after normalizing the contrast, bluring, then edge detecting.

@param img Input Image

@returns Resulting image after filtering pipeline
'''
def applyFilters(img):
    scale = 4
    blur_size = 21
    size = 3
    
    # Apply Linear Stretching
    image_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    stretched_yuv = image_yuv
    stretched_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    stretched = cv2.cvtColor(stretched_yuv, cv2.COLOR_YUV2BGR)
        
    # Apply Gausian Filter
    lowPass = cv2.GaussianBlur(stretched, (blur_size, blur_size),0)
    
    # Apply Sobel Filter
    b, g, r = cv2.split(lowPass)
    bx = cv2.Sobel(b, cv2.CV_8U, 1,0, ksize=size)
    by = cv2.Sobel(b, cv2.CV_8U, 0,1, ksize=size)
    b = cv2.addWeighted(bx, 0.5, by, 0.5, 0)
    gx = cv2.Sobel(g, cv2.CV_8U, 1,0, ksize=size)
    gy = cv2.Sobel(g, cv2.CV_8U, 0,1, ksize=size)
    g = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    rx = cv2.Sobel(r, cv2.CV_8U, 1,0, ksize=size)
    ry = cv2.Sobel(r, cv2.CV_8U, 0,1, ksize=size)
    r = cv2.addWeighted(rx, 0.5, ry, 0.5, 0)
    sobel = cv2.merge([r,g,b])
    return sobel


# If run then it will use a test image and display the results at each stage of the pipeline
def main():
    # Parameters
    # scale = 4
    blur_size = 51
    size = 3

    # Read in image
    jsImag = cv2.imread("./test/imag/tigerWoods.jpg")
    (w, h) = jsImag.shape[:2]
    scale = 512 / w
    dim = (int(w * scale), int(h * scale))
    jsImag_res = cv2.resize(jsImag, dim)
    cv2.imshow(('Original'),jsImag_res)
    
    
    # Apply Linear Stretching
    image_yuv = cv2.cvtColor(jsImag, cv2.COLOR_RGB2YCrCb)
    stretched_yuv = image_yuv
    stretched_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    stretched = cv2.cvtColor(stretched_yuv, cv2.COLOR_YUV2BGR)
    
    stretched_res = cv2.resize(stretched, (int(h/scale), int(w/scale)))
    cv2.imshow(('Stretched'),stretched_res)
    
    # Apply Gausian Filter
    lowPass = cv2.GaussianBlur(stretched, (blur_size, blur_size),0)
    lowPass_res = cv2.resize(lowPass, (int(h/scale), int(w/scale)))
    cv2.imshow(('Gaussian'),lowPass_res)
    
    # Apply Sobel Filter
    b, g, r = cv2.split(lowPass)
    bx = cv2.Sobel(b, cv2.CV_8U, 1,0, ksize=size)
    by = cv2.Sobel(b, cv2.CV_8U, 0,1, ksize=size)
    b = cv2.addWeighted(bx, 0.5, by, 0.5, 0)
    gx = cv2.Sobel(g, cv2.CV_8U, 1,0, ksize=size)
    gy = cv2.Sobel(g, cv2.CV_8U, 0,1, ksize=size)
    g = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    rx = cv2.Sobel(r, cv2.CV_8U, 1,0, ksize=size)
    ry = cv2.Sobel(r, cv2.CV_8U, 0,1, ksize=size)
    r = cv2.addWeighted(rx, 0.5, ry, 0.5, 0)
    sobel = cv2.merge((b,g,r))
    sobel_res = cv2.resize(sobel, (int(h/scale), int(w/scale)))
    cv2.imshow(('Sobel'),sobel_res)
    
    cv2.waitKey(0)

if __name__ == "__main__":
    main()