import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle as pkl
from moviepy.editor import VideoFileClip


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

# draw a polygon on an image
def my_draw_polygon(img,vertices):
    for ii in range(len(vertices)):
        ind1 = (ii+1) % len(vertices)
        v0 = vertices[ii]
        v1 = vertices[ind1]
        draw_lines(img, [[(v0[0],v0[1],v1[0],v1[1])]])

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
    # plt.title("hsv combined")
    # plt.colorbar()
    return hsv_binary

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
    # plt.title("rgb combined")
    # plt.colorbar()
    return rgb_binary

def colorThreshold(img):
    hsv = col_hsvthresh(img)
    rgb = col_rgbthresh(img)
    combined = np.zeros_like(hsv)
    combined[ (hsv==1) & (rgb == 1) ] = 1
    return combined


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

#########################################################
# A. Calibrate the Camera: Find the parameters to correct for image distortion
#########################################################

compute_calibration = False

if compute_calibration:
    print('Calibrating camera ...')
    nx = 9
    ny = 6
    chessimgs = glob.glob('./camera_cal/*.jpg')

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    for chessimg in chessimgs:
        img = cv2.imread(chessimg)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        img_size = (gray.shape[1], gray.shape[0])

        if ret == True:
            # If we found corners, draw them! (just for fun)
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(250)
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print("Saving distortion coefficients ...")
    pkl.dump( (ret, mtx, dist, rvecs, tvecs) , open( "distortion.p", "wb" ) )
else:
    print("Loading distortion coefficients ...")
    ret, mtx, dist, rvecs, tvecs = pkl.load( open( "distortion.p", "rb" ) )

# Plot the results of distortion correction
# chessimg = cv2.imread('./camera_cal/calibration1.jpg')
# uchessimg = cv2.undistort(chessimg, mtx, dist, None, mtx)
# cv2.imshow('chessimg', chessimg)
# cv2.imshow('uchessimg', uchessimg)ret, mtx, dist, rvecs, tvecs = 
# cv2.waitKey(0)

############################################################
# B. Determine the perspective transform 
############################################################
imgrect = mpimg.imread("./test_images/straight_lines1.jpg")
imx = imgrect.shape[1]
imy = imgrect.shape[0]

warpedrectvertices = np.array([
    [305,650],
    [525, 500], 
    [760, 500], 
    [1000,650]], dtype= np.float32)
offset = 160
rectvertices = np.array([
    [offset, imy-offset],
    [offset, offset],
    [imx-offset, offset],
    [imx-offset, imy-offset]], dtype = np.float32)
M = cv2.getPerspectiveTransform(warpedrectvertices, rectvertices)
Minv = cv2.getPerspectiveTransform(rectvertices, warpedrectvertices)

warped = cv2.warpPerspective(imgrect, M, (imx, imy))

# Plot the perspecetive transform
# plt.figure(10)
# plt.imshow(warped)

# my_draw_polygon(imgrect,warpedrectvertices[0])
# plt.figure(28)
# plt.imshow(imgrect)
# plt.show()


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def findLaneLines(pristine,binary_warped):

    # Take a histogram of the (bottom) half of the image
    histogram = np.sum(binary_warped, axis=0)

    # plt.figure(80)
    # plt.plot(histogram)
    # plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.figure(30)
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

        # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (pristine.shape[1], pristine.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(pristine, 1, newwarp, 0.3, 0)

    return(result)

#####################################################
# Set pipeline parameters
#####################################################
# Choose a Sobel kernel size
kernelsize = 5 # Choose a larger odd number to smooth gradient measurements
makeplots = False
##########################################################
# Define the main image processing pipeline
##########################################################
def pipeline(image):

    ####################################################
    # Input: img (image of the roadway ahead)
    # Returns: result (image of the roadway ahead with the lane annotated)
    #####################################################################

    #####################################################
    # 0. Get information about this image and copy the image.
    ######################################################
    imx = image.shape[1]
    imy = image.shape[0]
    img = np.copy(image)

    # if makeplots:
    #     plt.figure(100)
    #     plt.imshow(img)
    #     plt.title("Original Image")

    #####################################################
    # 1. Use camera calibration to remove distortion
    ####################################################
    img = cv2.undistort(img, mtx, dist, None, mtx)
    undistorted = np.copy(img)

    # if makeplots:
    #     plt.figure(101)
    #     plt.imshow(img)
    #     plt.title("Distortion Correction")

    ######################################################
    # 2. Get warped perspective
    ######################################################
    img = cv2.warpPerspective(img, M, (imx, imy))

    if makeplots:
        plt.figure(102)
        plt.imshow(img)
        plt.title("Warped Perspective")

    #######################################################
    # 3. Consider only the region of interest 
    #######################################################   
    dx = 0
    dybot = 145
    ROIvertices = np.array([[(dx,imy-dybot),(dx, 0), (imx-dx, 0), (imx-dx,imy-dybot)]], dtype=np.int32)
    img = region_of_interest(img,ROIvertices)

    # if makeplots:
    #     plt.figure(103)
    #     plt.imshow(img)
    #     plt.title("Cropped to Region of Interest")

        # imgpoly = np.copy(img)
        # draw_polygon(imgpoly,ROIvertices)
        # plt.figure(18)
        # plt.imshow(imgpoly)

    #######################################################
    # 4. Color and gradient thresholding in HLS
    ########################################################
    # combined = gradientThreshold(img,kernelsize)
    colpixels = colorThreshold(img)
    scaled_col = np.uint8(255*colpixels)
    if makeplots:
        plt.figure(304)
        plt.imshow(scaled_col)
        plt.title("Color Thresholded")

    grpixels = gradientThreshold(img,kernelsize)
    scaled_gr = np.uint8(255*grpixels)
    if makeplots:
        plt.figure(305)
        plt.imshow(scaled_gr)
        plt.title("Gradient Thresholded")

    combined = np.zeros_like(colpixels)
    combined[ (colpixels==1) & (grpixels == 1) ] = 1

    scaled_combined = np.uint8(255*combined)


    if makeplots:
        plt.figure(104)
        plt.imshow(scaled_combined)
        plt.title("Thresholded")

    ##########################################################
    # 5. Find lane lines 
    ##########################################################
    llimg = findLaneLines(undistorted,combined)
    
    if makeplots:
        plt.figure(105)
        plt.imshow(llimg)
        plt.title("Polynomial Fit")

    return llimg
    

if __name__ == '__main__':

    processimages = False   

    processvideos = True

#########################################################################
# Process images
#########################################################################

    if processimages:
        # imagenames = glob.glob('./test_images/*.jpg')
        imagenames = ["./frames/frame_0363.jpg"]
        # imagenames = ["./test_images/test4.jpg"]

        for imagename in imagenames:

            print(imagename)

            image = mpimg.imread(imagename)
            
            result = pipeline(image)

            # Plot the result
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            f.tight_layout()

            ax1.imshow(image)
            ax1.set_title('Original Image : %s' % imagename, fontsize=12)

            ax2.imshow(result)
            ax2.set_title('Pipeline Result', fontsize=12)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


#########################################################################
# Process videos
#########################################################################
    
    if processvideos:

        videos = ["project_video.mp4"]
        for video in videos:            # process each video in the directory
            print("processing {}".format(video))
            processed_video = "ann_" + video
            clip1 = VideoFileClip(video)
            processed_clip = clip1.fl_image(lambda img: pipeline(img)) # run the lane lines processor
            get_ipython().run_line_magic('time', 'processed_clip.write_videofile(processed_video, audio=False)')    # save the output

    print("DONE!!!")
    plt.show()























###########################################################################################
###########################################################################################
###########################################################################################


# def findLaneLines(warped):

#     window_width = 50 
#     window_height = 80 # Break image into 9 vertical layers since image height is 720
#     margin = 100 # How much to slide left and right for searching

#     window_centroids = find_window_centroids(warped, window_width, window_height, margin)

#     # If we found any window centers
#     if len(window_centroids) > 0:

#         print(window_centroids)

#         # Points used to draw all the left and right windows
#         l_points = np.zeros_like(warped)
#         r_points = np.zeros_like(warped)

#         # Go through each level and draw the windows    
#         for level in range(0,len(window_centroids)):
#             # Window_mask is a function to draw window areas
#             l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
#             r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
#             # Add graphic points from window mask here to total pixels found 
#             l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
#             r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

#         # Draw the results
#         template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
#         zero_channel = np.zeros_like(template) # create a zero color channel
#         template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
#         warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
#         output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
#     # If no window centers found, just display orginal road image
#     else:
#         output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

#     # Display the final results
#     plt.imshow(output)
#     plt.title('window fitting results')
#     plt.show()
if(False):




    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import glob
    import cv2

    # Read in a thresholded image
    warped = mpimg.imread('warped_example.jpg')
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
    	    # convolve the window into the vertical slice of the image
    	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
    	    conv_signal = np.convolve(window, image_layer)
    	    # Find the best left centroid by using past left center as a reference
    	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    	    offset = window_width/2
    	    l_min_index = int(max(l_center+offset-margin,0))
    	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
    	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    	    # Find the best right centroid by using past right center as a reference
    	    r_min_index = int(max(r_center+offset-margin,0))
    	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
    	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    	    # Add what we found for that layer
    	    window_centroids.append((l_center,r_center))

        return window_centroids

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()


    import numpy as np
    import matplotlib.pyplot as plt
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                  for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the 


    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m