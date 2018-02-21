##############################################################
# We'll need these libraries
import cv2
import numpy as np
import os
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

mode = 0
if(mode==0):
    saveplots = False
else:
    saveplots = False

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
if(saveplots):
    chessimg = cv2.imread('./camera_cal/calibration1.jpg')
    uchessimg = cv2.undistort(chessimg, mtx, dist, None, mtx)
    cv2.imwrite('output_images/chessimg.png',chessimg)
    cv2.imwrite('output_images/uchessimg.png',uchessimg)

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

    if(saveplots):
        plt.figure(100)
        plt.imshow(img)
        plt.title("Original Image")
        plt.savefig("output_images/original_image.png")
        # plt.close()

    #####################################################
    # 1. Use camera calibration to remove distortion
    ####################################################
    img = cv2.undistort(img, mtx, dist, None, mtx)
    undistorted = np.copy(img)      # save this image for later, so that we can use it for annotating

    if(saveplots):
        plt.figure(101)
        plt.imshow(img)
        plt.title("Distortion Correction")
        plt.savefig("output_images/distortion_correction.png")
        # plt.close()

    ######################################################
    # 2. Get warped perspective
    ######################################################
    img = cv2.warpPerspective(img, M, (imx, imy))

    if(saveplots):
        plt.figure(102)
        plt.imshow(img)
        plt.title("Warped Perspective")
        plt.savefig("output_images/warped_perspective.png")
        # plt.close()


    ############################################################
    # 3. Consider only the region of interest (remove the hood)
    ############################################################   
    # dybot = 588     # cut pixels below y=dybot (the hood)
    # ROIvertices = np.array([[(0,dybot),(0, 0), (imx, 0), (imx,740)]], dtype=np.int32)
    # img = ut.region_of_interest(img,ROIvertices)

    if(saveplots):
        plt.figure(103)
        plt.imshow(img)
        plt.title("Cropped to Region of Interest")
        plt.savefig("output_images/cropped.png")
        # plt.close()
        # plt.show()


    #######################################################
    # 4. Color and gradient thresholding in HLS and RGB
    ########################################################
    combined_binary = ut.colgradientThresholding(img)
    scaled_combined = np.uint8(255*combined_binary)

    if(saveplots):
        plt.figure(105)
        plt.imshow(scaled_combined)
        plt.title("Thresholded Image")
        plt.savefig("output_images/thresholded.png")
        # plt.close()


    ##########################################################
    # 5. Find lane lines and return
    ##########################################################
    llimg = ll.findLaneLines(undistorted,combined_binary,Minv)
    if(saveplots):
        plt.figure(106)
        plt.imshow(llimg)
        plt.title("Polynomial Fit")
        plt.savefig("output_images/polynomialfit.png")
        # plt.close()

    return llimg
    
#######################################################################
# MAIN
#######################################################################

if __name__ == '__main__':

    

# Let's choose what we want to do (process still images or video?)
    if(mode==0):
        processimages = True
        processvideos = False
    else:
        processimages = False
        processvideos = True

#########################################################################
# Process images
#########################################################################

    if processimages:
        # imagenames = glob.glob('./test_images/*.jpg')
        imagenames = ['./test_images/test3.jpg']

        for imagename in imagenames:

            filename = os.path.basename(imagename)

            print(imagename)
            image = mpimg.imread(imagename)
            
            # run this image through the pipeline
            result = pipeline(image)

            # Plot the result
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title('Original Image : %s' % filename, fontsize=12)
            ax2.imshow(result)
            ax2.set_title('Pipeline Result', fontsize=12)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig("output_images/ann_%s" % filename)
            # plt.close()


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
    