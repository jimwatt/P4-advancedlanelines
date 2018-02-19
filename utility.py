import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    # for ii in range(3):
    #     plt.figure(800+ii)
    #     plt.imshow(hsv[:,:,ii])
    #     plt.title("hsv %d" % ii)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    hsv_binary = np.zeros_like(v)
    hsv_binary[v >= 50 ] = 1
    # plt.figure(803)
    # plt.imshow(np.uint8(hsv_binary)*255)
    # plt.title("HSV combined")
    # plt.savefig("output_images/hsv_combined.png")
    # plt.close()
    return hsv_binary

# threshold an image using rgb values
def col_rgbthresh(rgb):
    # for ii in range(3):
    #     plt.figure(900+ii)
    #     plt.imshow(rgb[:,:,ii])
    #     plt.title("rgb %d" % ii)
    #     plt.colorbar()
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    rgb_binary = np.zeros_like(r)
    rgb_binary[(r >= 80) 
                | (g >= 80)  
                | (b >= 80) ] = 1  
    # plt.figure(903)
    # plt.imshow(np.uint8(rgb_binary)*255)
    # plt.title("RGB combined")
    # plt.savefig("output_images/rgb_combined.png")
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
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=kernelsize, thresh=(25, 255))
    grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=kernelsize, thresh=(25, 255))
    mag_binary = mag_thresh(s_channel, sobel_kernel=kernelsize, thresh=(6, 255))
    slope = 0.4
    dir_binary = dir_thresh(s_channel, sobel_kernel=kernelsize, thresh=(-slope*np.pi, slope*np.pi))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

# combine the results of color and gradient thresholding
def colgradientThresholding(img):

    kernelsize = 5
    
    colpixels = colorThreshold(img)
    scaled_col = np.uint8(255*colpixels)
    
    # plt.figure(304)
    # plt.imshow(scaled_col)
    # plt.title("Color Thresholded")
    # plt.savefig("output_images/color_threshold.png")
    # plt.close()

    grpixels = gradientThreshold(img,kernelsize)
    scaled_gr = np.uint8(255*grpixels)
    
    # plt.figure(305)
    # plt.imshow(scaled_gr)
    # plt.title("Gradient Thresholded")
    # plt.savefig("output_images/gradient_threshold.png")
    # plt.close()

    combined_binary = np.zeros_like(colpixels)
    combined_binary[ (colpixels==1) & (grpixels == 1) ] = 1

    return combined_binary