## Advanced Lane Line Detection

[//]: #	"Image References"
[image1]: ./output_images/ann_test6.jpg	"Example Output"
[image2]: ./output_images/chessimg.png	"Distorted Image"
[image3]: ./output_images/uchessimg.png	"Corrected Image after Camera Calibration"
[image4]: ./output_images/original_image.png	"Original Image"
[image5]: ./output_images/distortion_correction.png	"Distortion Correction "
[image6]: ./output_images/perspective.png	"Perspective"
[image7]: ./output_images/birdseye.png	"Birds-Eye"
[image8]: ./output_images/cropped.png	"Cropped"
[image9]: ./output_images/color_threshold.png	"Color Thresholded"
[image10]: ./output_images/gradient_threshold.png	"Gradient Thresholded"
[image11]: ./output_images/thresholded.png	"Thresholded"
[image12]: ./output_images/ann_straight_lines1.jpg	"straight_lines1"
[image13]: ./output_images/ann_straight_lines2.jpg	"straight_lines2"
[image14]: ./output_images/ann_test1.jpg	"test1"
[image15]: ./output_images/ann_test2.jpg	"test2"
[image16]: ./output_images/ann_test3.jpg	"test3"
[image17]: ./output_images/ann_test4.jpg	"test4"
[image18]: ./output_images/ann_test5.jpg	"test5"
[image19]: ./output_images/ann_test6.jpg	"test6"
[image20]: ./output_images/rgb_combined	"rgb"
[image21]: ./output_images/hsv_combined	"hsv"
[image22]: ./output_images/color_threshold.png	"color"
[image23]: ./output_images/grad_combined.png	"grad"
[image24]: ./output_images/searching.png	"searching"
[image25]: ./output_images/histogram.png	"histogram"
[image26]: ./output_images/polynomialfit.png	"polynomialfit"
[video1]: ./ann_project_video.mp4	"Video"

### Goal:

Given a video stream containing images of the road ahead, use the painted lane markings to annotate the lane ahead. 

Example output is shown here:

![alt text][image1]

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* **Camera Calibration** : Correct the images for camera distortion.
* **Perspective Warping** : Apply perspective warping to obtain a "bird's-eye" view of the road ahead.
* **Image Cropping** : Crop the image to the region of interest.
* **Pixel Thresholding** : Use color and gradient thresholding to detect pixels corresponding to lane line markings.
* **Peak Finding** : Use simple windowing to determine location of the left and right lane lines at multiple locations in the road ahead.
* **Curve Fitting** : Fit the left and right lane line locations with a quadratic polynomial.   
* **Lane Geometry** : From the polynomial fit, determine the radius of curvature of each lane line, and the offset from center of the car in the lane.
* **Lane Annotation** : Map the fitted lane lines to the original image and annotate the image.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup 

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.    

You're reading it!

The source code for this project is found in the following files:

1. **p4code.py** : The main entry point for the code.  Run the code by entering `python p4code.py`
2. **utility.py** : Various general purpose helper functions for working with images, including color and gradient thresholding.
3. **calibrate.py** : Functions for performing camera calibration using images of chessboards to correct for distortion.
4. **perspective.py** : Functions for performing image perspective transformations.  This is how we achieve the birds eye view of the road.
5. **lanelines.py** : Functions for extracting lane lines from a previously thresholded image.  The thresholded image should try to retain only those pixels that are part of the lane markings.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrateCamera()` function definition in **calibrate.py** (lines 2-33).  

* I start by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. I constrain the rectified chessboard corners to be fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj` is just a fixed array of coordinates, and `objpoints` will be appended with a copy of `obj` every time I successfully detect all chessboard corners in a test image. 
* The list `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection (for each of the chessboard images in `./camer_cal`).  The `imgpoints` are detected using the openCV function `cv2.findChessBoardCorners()`.  Note that I used the `cv2.cornerSubPix()` routine (that uses an iterative solver) to further refine the corner locations within the image to subpixel resolution.
* I then used `objpoints` and `imgpoints` as source and destination pairs to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Applying camera calibration corrections to the following image:

![alt text][image2]

yields the following corrected image:

![alt text][image3]

### Pipeline (single images)

We now describe the steps in the image processing pipeline.  The pipeline is defined in lines 54-135 in `p4code.py`.  

#### 1. Provide an example of a distortion-corrected image.

After having obtained the image correction coefficients from the chessboard images, we can apply these corrections to the images in the video stream.  For example, here is a typical image:

![alt text][image4]

After applying the distortion corrections, we obtain this (very similar looking) image:

![alt text][image5]

In the code, the distortion correction step is applied at line 74 in the code.

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `getPerspectiveTransforms()`, which appears in lines 5 through 40 in the file `perspective.py` .  The `getPerspectiveTransforms()` function defines `warpedrectvertices` as the pixel locations of an apparent rectangle in a perspective image (in which the lane lines are relatively straight so that the only warping is due to perspective --- not actual curvature), as well as destination points `rectvertices` which are hard-coded as the corners of a rectangle seen directly from above without perspective.   I chose to hard-code the source and destination points in the following manner: 

```
# As determined from an image with relatively straight lane lines.
warpedrectvertices = np.array([
	    [305,650],
	    [525, 500], 
	    [760, 500], 
	    [1000,650]], dtype= np.float32)
```

```python
offsetx = 160
offsety = 60
imx = imgrect.shape[1]
imy = imgrect.shape[0]
rectvertices = np.array([
	    [offsetx, imy-offsety],
	    [offsetx, offsety],
	    [imx-offsetx, offsety],
	    [imx-offsetx, imy-offsety]], dtype = np.float32)
```

This resulted in the following source and destination points:

| Source    | Destination |
| --------- | ----------- |
| 305, 650  | 160, 660    |
| 525, 500  | 160, 60     |
| 760, 500  | 1120, 60    |
| 1000, 650 | 1120, 660   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

**Source:**

![alt text][image6]

**Destination:**

![alt text][image7]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 118 through 143 in `utility.py`).  

I found that neither color thresholding alone, nor gradient thresholding alone was sufficient to detect the lane markings along the entire length of the road.

For the following (warped) original image:

![alt text][image8]

![alt text][image9]![alt text][image10]

![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In lines 142 through 155 in `lanelines.py`, I computed the radius of curvature and offset from the center of the lane.

16. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./ann_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
