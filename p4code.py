##############################################################
# We'll need these libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle as pkl
from moviepy.editor import VideoFileClip

############################################################################
# Import some additional helper functions
import utility as ut # general purpose utility and helper functions
import lanelines as ll # all tools for extracting lanelines from a thresholded image
import calibrate as cal # camera calibration functionality
import perspective as ps # perspective transformation (warping) functionality
#############################################################################

##################################################################################################
# A. CAMERA CALIBRATION: Calibrate the Camera: Find the parameters to correct for image distortion
##################################################################################################

# Do we want to recompute the camera calibration, or just use saved values from last time
compute_calibration = False
if compute_calibration:
    ret, mtx, dist, rvecs, tvecs = cal.calibrateCamera()
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

############################################################################################################
# B. PERSPECTIVE TRANSFORM: Determine the perspective transform for creating overhead view (and its inverse)
############################################################################################################

M,Minv = ps.getPerspectiveTransforms()

##########################################################
# PIPELINE: Define the main image processing pipeline
# Input: img (image of the roadway ahead)
# Returns: result (image of the roadway ahead with the lane annotated)
##########################################################
def pipeline(image):

    #####################################################
    # 0. Get information about this image and copy the image.
    ######################################################
    imx = image.shape[1]
    imy = image.shape[0]
    img = np.copy(image)

    # plt.figure(100)
    # plt.imshow(img)
    # plt.title("Original Image")

    #####################################################
    # 1. Use camera calibration to remove distortion
    ####################################################
    img = cv2.undistort(img, mtx, dist, None, mtx)
    undistorted = np.copy(img)      # save this image for later, so that we can use it for annotating

    # plt.figure(101)
    # plt.imshow(img)
    # plt.title("Distortion Correction")

    ######################################################
    # 2. Get warped perspective
    ######################################################
    img = cv2.warpPerspective(img, M, (imx, imy))

    # plt.figure(102)
    # plt.imshow(img)
    # plt.title("Warped Perspective")

    ############################################################
    # 3. Consider only the region of interest (remove the hood)
    ############################################################   
    dybot = 145     # cut off the bottom dybot pixels (the hood)
    ROIvertices = np.array([[(0,imy-dybot),(0, 0), (imx, 0), (imx,imy-dybot)]], dtype=np.int32)
    img = ut.region_of_interest(img,ROIvertices)

    # plt.figure(103)
    # plt.imshow(img)
    # plt.title("Cropped to Region of Interest")

    # imgpoly = np.copy(img)
    # draw_polygon(imgpoly,ROIvertices)
    # plt.figure(104)
    # plt.imshow(imgpoly)

    #######################################################
    # 4. Color and gradient thresholding in HLS and RGB
    ########################################################
    combined_binary = ut.colgradientThresholding(img)
    scaled_combined = np.uint8(255*combined_binary)

    # plt.figure(105)
    # plt.imshow(scaled_combined)
    # plt.title("Thresholded")

    ##########################################################
    # 5. Find lane lines and return
    ##########################################################
    llimg = ll.findLaneLines(undistorted,combined_binary,Minv)
    
    # plt.figure(106)
    # plt.imshow(llimg)
    # plt.title("Polynomial Fit")

    return llimg
    
#######################################################################
# MAIN
#######################################################################

if __name__ == '__main__':

# Let's choose what we want to do (process still images or video?)
    processimages = True   
    processvideos = False

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
            
            # run this image through the pipeline
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
        for video in videos:           
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