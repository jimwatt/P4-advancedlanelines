
import numpy as np
import cv2
import matplotlib.pyplot as plt

saveplots = False

#################################################################################'
# Functionality for finding lanelines
#################################################################################

# Given an image, return the image with all pixels set to zero except those in teh specified window cell.
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Given a y value in pixels, a quadtratic fit in pixels, and scale factors xmpp, and ympp, compute radius of curvature in meters
def calculateCurveRad(y,fit,xmpp,ympp):
    A = fit[0]*xmpp/(ympp*ympp)
    B = fit[1]*xmpp/ympp
    return (1.0 + ((2.0*A*y + B)**2)**(1.5)) / np.abs(2*A)


# Given the binary warped image containing pixels corresponding to the lane lines, compute a polynomial fit to the lanelines, and draw them on the pristine image.
def findLaneLines(pristine,binary_warped, Minv):

    # Take a histogram of the (bottom) half of the image
    histogram = np.sum(binary_warped, axis=0)

    if(saveplots):
        plt.figure(80)
        plt.plot(histogram)
        plt.title("Histogram of Laneline Pixels")
        plt.savefig("output_images/histogram.png")
        # plt.close()

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
    margin = 150
    # Set minimum number of pixels found to recenter window
    minpix = 20
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
    lpoly = np.poly1d(left_fit)
    right_fit = np.polyfit(righty, rightx, 2)
    rpoly = np.poly1d(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fitx = lpoly(ploty)
    right_fitx = rpoly(ploty)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if(saveplots):
        plt.figure(30)
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig("output_images/searching.png")
        # plt.close()

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

    ###############################################################################S
    # Determine radius of curvature, and offset from center, and write on the image

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

     # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = calculateCurveRad(y_eval,left_fit,xm_per_pix,ym_per_pix)
    right_curverad = calculateCurveRad(y_eval,right_fit,xm_per_pix,ym_per_pix)

    # Compute offset from lane center
    offcenter = -( 0.5*(lpoly(y_eval)+rpoly(y_eval)) - 0.5*binary_warped.shape[1] ) * xm_per_pix

    cv2.putText(img=result,text="Left  : %.1f km" % (left_curverad/1000.0), org=(20,110), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.5, color=(255,255,255), thickness=3)
    cv2.putText(img=result,text="Right : %.1f km" % (right_curverad/1000.0), org=(20,170), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.5, color=(255,255,255), thickness=3)
    cv2.putText(img=result,text="Offset : %d cm" % np.int(offcenter*100.0), org=(20,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.5, color=(255,255,255), thickness=3)
    
    return(result)