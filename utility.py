import numpy as np
import cv2
import matplotlib.pyplot as plt

saveplots = False
##########################################################################
# Define some general purpose utility helper functions
##########################################################################

# Mask an image by the polygon defined in vertices
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask,  vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# draw lines on an image
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# compute the absolute value of gradient in x or y direction
def abs_sobel_thresh(image, orient, sobel_kernel, thresh):
    if(orient=='x'):
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    elif(orient=='y'):
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    abs_sobel = np.absolute(sobel) 
    abs_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sbinary = np.zeros_like(abs_sobel)
    sbinary[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1
    return sbinary

# compute the magnitude of the gradient
def mag_thresh(image, sobel_kernel, thresh):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    mag_sobel = np.sqrt(sobelx**2 + sobely**2) 
    mag_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    mag_binary = np.zeros_like(mag_sobel)
    mag_binary[(mag_sobel >= thresh[0]) & (mag_sobel <= thresh[1])] = 1
    return mag_binary

# compute the direction of the gradient
def dir_thresh(image, sobel_kernel, thresh):
    sobelx = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)) # Take the derivative in x
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    dir_sobel = np.arctan2(sobely,sobelx)
    dir_binary = np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return dir_binary

# threshold an image using hsv values
def col_hsvthresh(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    hsv_binary = np.zeros_like(v)
    hsv_binary[(h<60) | (h<50) | (v >=130) ] = 1
    if(saveplots):
        plt.figure(800)
        for ii in range(3):
            plt.subplot(2,2,ii+1)
            plt.imshow(hsv[:,:,ii])
            plt.title("hsv %d" % ii)
            plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(np.uint8(hsv_binary)*255)
        plt.title("HSV combined")
        plt.savefig("output_images/hsv_combined.png")
        # plt.close()
    return hsv_binary

# threshold an image using rgb values
def col_rgbthresh(rgb):
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    rgb_binary = np.zeros_like(r)
    rgb_binary[(r >= 140) 
                | (g >= 90)  
                | (b >= 140) ] = 1
    if(saveplots):
        plt.figure(900)
        for ii in range(3):
            plt.subplot(2,2,ii+1)
            plt.imshow(rgb[:,:,ii])
            plt.title("rgb %d" % ii)
            plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(np.uint8(rgb_binary)*255)
        plt.title("RGB combined")
        plt.savefig("output_images/rgb_combined.png")
        # plt.close()
    return rgb_binary

# threshold an image on color using rgb and hasv spaces
def colorThreshold(img):
    hsv = col_hsvthresh(img)
    rgb = col_rgbthresh(img)
    combined = np.zeros_like(hsv)
    combined[ (hsv==1) & (rgb == 1) ] = 1
    return combined

# threshold an rgb image using gradients in the s channel
def gradientThreshold(img, kernelsize):
    imy = img.shape[0]
    imx = img.shape[1]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=kernelsize, thresh=(8, 255))
    grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=kernelsize, thresh=(8, 255))
    mag_binary = mag_thresh(s_channel, sobel_kernel=kernelsize, thresh=(5, 255))
    slope = 0.5
    dir_binary = dir_thresh(s_channel, sobel_kernel=kernelsize, thresh=(-slope*np.pi, slope*np.pi))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # Remove hood points with spatial filter
    combined[650:imy,350:1000] = 0
    vertices = np.array([[(320,imy),(320, 0), (930, 0), (930,imy)]], dtype=np.int32)
    cv2.fillPoly(combined,  vertices, 0)

    if(saveplots):
        plt.figure(1300)
        plt.subplot(3,2,1)
        plt.imshow(np.uint8(gradx)*255)
        plt.title("gradx")
        plt.colorbar()
        plt.subplot(3,2,2)
        plt.imshow(np.uint8(grady)*255)
        plt.title("grady")
        plt.colorbar()
        plt.subplot(3,2,3)
        plt.imshow(np.uint8(mag_binary)*255)
        plt.title("mag_binary")
        plt.colorbar()
        plt.subplot(3,2,4)
        plt.imshow(np.uint8(dir_binary)*255)
        plt.title("dir_binary")
        plt.savefig("output_images/grad_combined.png")
        plt.subplot(3,2,5)
        plt.imshow(np.uint8(combined)*255)
        plt.title("grad combined")
        plt.savefig("output_images/grad_combined.png")
        # plt.close()
    return combined

# combine the results of color and gradient thresholding
def colgradientThresholding(img):

    kernelsize = 3
    
    colpixels = colorThreshold(img)
    scaled_col = np.uint8(255*colpixels)
    
    if(saveplots):
        plt.figure(304)
        plt.imshow(scaled_col)
        plt.title("Color Thresholded")
        plt.savefig("output_images/color_threshold.png")
        # plt.close()

    grpixels = gradientThreshold(img,kernelsize)
    scaled_gr = np.uint8(255*grpixels)
    
    if(saveplots):
        plt.figure(305)
        plt.imshow(scaled_gr)
        plt.title("Gradient Thresholded")
        plt.savefig("output_images/gradient_threshold.png")
        # plt.close()

    combined_binary = np.zeros_like(colpixels)
    combined_binary[ (colpixels==1) & (grpixels == 1) ] = 1

    if(saveplots):
        plt.figure(306)
        plt.imshow(np.uint8(combined_binary)*255)
        plt.title("Combined Threshold")
        plt.savefig("output_images/combined_threshold.png")
        # plt.close()

    return combined_binary