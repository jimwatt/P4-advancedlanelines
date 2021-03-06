# Advanced Lane Lines

![alt text][image18]

This project is in conjunction with the Udacity Self-Driving Car course.  In particular, the goal of this project is to use classical computer vision techniques to robustly detect lane lines and describe their geometry.  Given an input video of the road ahead, this code will annotate the lane ahead in each frame of the video. The main steps are: 

* Correct the images for camera distortion.

* Apply perspective warping to obtain a "bird's-eye" view of the road ahead.

* Crop to the region of interest.

* Use color and gradient thresholding to detect pixels corresponding to lane line markings.

* Use simple windowing to determine location of the left and right lane lines at multiple locations in the road ahead.

* Fit the left and right lane lane locations with a quadratic polynomial.   

* From the polynomial fit, determine the radius of curvature of each lane line, and the offset from center of the car in the lane.

* More detail about the lane finding approach is provided in writeup.md.


## Getting Started

### Prerequisites

Imported packages include:

```
Python
cv2
matplotlib
moviepy
glob
```

### Installing

No install is required --- simply clone this project from GitHub:

```
git clone https://github.com/jimwatt/P4-advancedlanelines.git
```

## Running the Code

* The starting point for the code is p4code.py.  In the top directory, run 

  `python p4code.py`

  Consider changing flags in main() to switch between processing video and still images.


##Files##

The source code for this project is found in the following files:

1. **p4code.py** : The main entry point for the code.  Run the code by entering `ipython p4code.py`
2. **utility.py** : Various general purpose helper functions for working with images, including color and gradient thresholding.
3. **calibrate.py** : Functions for performing camera calibration using images of chessboards to correct for distortion.
4. **perspective.py** : Functions for performing image perspective transformations.  This is how we achieve the birds eye view of the road.
5. **lanelines.py** : Functions for extracting and annotating lanelines from a previously thresholded image.  The thresholded image should try to retain only those pixels that are part of the lane markings.  



## Authors

* **James Watt**

<!--## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details-->

## Acknowledgments
This project is a submission to the Udacity Self-Driving Car nanodegree:

* https://github.com/udacity/CarND-Advanced-Lane-Lines.git

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
[image20]: ./output_images/rgb_combined.png	"rgb"
[image21]: ./output_images/hsv_combined.png	"hsv"
[image22]: ./output_images/color_threshold.png	"color"
[image23]: ./output_images/grad_combined.png	"grad"
[image24]: ./output_images/searching.png	"searching"
[image25]: ./output_images/histogram.png	"histogram"
[image26]: ./output_images/polynomialfit.png	"polynomialfit"
[video1]: ./ann_project_video.mp4	"Video"

### Goal:

Given a video stream containing images of the road ahead, use the painted lane markings in the image to detect and annotate the lane ahead.  Also, determine the geometry of the lane including radii of curvature and the position of the vehicle in the lane.

Example of the final output is shown here:

![alt text][image1]

------

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

- **Camera Calibration** : Correct the images for camera distortion.
- **Perspective Warping** : Apply perspective warping to obtain a "bird's-eye" view of the road ahead.
- **Pixel Thresholding** : Use color and gradient thresholding to detect pixels corresponding to lane line markings.
- **Peak Finding** : Use simple windowing to determine locations of the left and right lane lines at multiple locations in the road ahead.
- **Curve Fitting** : Fit the left and right lane line locations with a quadratic polynomial.   
- **Lane Geometry** : From the polynomial fit, determine the radius of curvature of each lane line, and the vehicle offset from the center of the lane.
- **Lane Annotation** : Map the fitted lane lines to the original image and annotate the image.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

------

### Writeup 

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.    

You're reading it!

The source code for this project is found in the following files:

1. **p4code.py** : The main entry point for the code.  Run the code by entering `ipython p4code.py`
2. **utility.py** : Various general purpose helper functions for working with images, including color and gradient thresholding.
3. **calibrate.py** : Functions for performing camera calibration using images of chessboards to correct for distortion.
4. **perspective.py** : Functions for performing image perspective transformations.  This is how we achieve the bird's eye view of the road.
5. **lanelines.py** : Functions for extracting lane lines from a previously thresholded image.  The thresholded image should try to retain only those pixels that are part of the lane markings.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrateCamera()` function definition in **calibrate.py** (lines 2-33).  

- I start by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. I constrain the rectified chessboard corners to be fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj` is just a fixed array of coordinates, and `objpoints` will be appended with a copy of `obj` every time I successfully detect all chessboard corners in a test image. 
- The list `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection (for each of the chessboard images in `./camer_cal`).  The `imgpoints` are detected using the openCV function `cv2.findChessBoardCorners()`.  Note that I used the `cv2.cornerSubPix()` routine (that uses an iterative solver) to further refine the corner locations within the image to subpixel resolution.
- I then used `objpoints` and `imgpoints` as source and destination pairs to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. 

For example, applying these camera calibration corrections to the following image:

![alt text][image2]

yields the following corrected image:

![alt text][image3]

### Pipeline (single images)

We now describe the steps in the image processing pipeline.  The pipeline is defined in lines 58-141 in `p4code.py`.  

#### 1. Provide an example of a distortion-corrected image.

After having obtained the image correction coefficients from the chessboard images, we can apply these corrections to the images in the video stream.  For example, here is a typical image:

![alt text][image4]

After applying the distortion corrections, we obtain this (very similar looking) image:

![alt text][image5]

In the code, the distortion correction step is applied at line 78 in the code.

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `getPerspectiveTransforms()`, which appears in lines 8 through 69 in the file `perspective.py` .  The `getPerspectiveTransforms()` function defines `warpedrectvertices` as the pixel locations of an apparent rectangle in a perspective image (in which the lane lines are relatively straight so that the only warping is due to perspective --- not actual curvature), as well as destination points `rectvertices` which are hard-coded as the corners of a rectangle seen directly from above without perspective.   I chose to hard-code the source and destination points in the following manner: 

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

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points on a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

**Source:**

![alt text][image6]

**Destination:**

![alt text][image7]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 150 through 184 in `utility.py`).  

I found that neither color thresholding alone, nor gradient thresholding alone was sufficient to detect the lane markings along the entire length of the road.

#### Color Thresholding:#### 

I used both **rgb** and **hsv** color spaces to detect the lane markings.

Consider the following original image in the warped bird's eye perspective:

![alt text][image8]

In the RGB color space, we apply thresholds on each channel to retain only the pixels (shown in yellow) in the fourth image in the following chart.

![alt text][image20]

Likewise, we apply thresholds to each of the three channels in the HSV color space, and retain only the pixels (again shown in yellow) in the following chart.

![alt text][image21]

Selecting only the points detected in both color spaces yields the following collection of pixels as candidates for the lane lines:

![alt text][image9]

#### Gradient Thresholding#### 

We also detect lane line pixels by computing and thresholding of the gradients in the S-channel of the HSV color space.  We apply thresholds on the size of the x and y gradients individually, as well as their magnitude.  

I did not find much additional benefit when including direction of the gradient.

The image below shows which pixels are detected using gradients.

![alt text][image23]

Using gradients alone to detect lane markings, yields the following result:![alt text][image10]

#### Combined Color and Gradient Thresholding####

Keeping only those pixels detected by both color and gradient thresholding yields the following detections for the lane lines.

![alt text][image11]

It remains to fit these pixels to determine the geometry of the left and right lane lines.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the approach provided in the Udacity course material to determine points along the lane line, and then fitting those points with a quadratic polynomial.

- First, we take a histogram of the detected pixels to identify base locations for both the right and left lines.  An example histogram is shown here: 



![alt text][image25]

- Then, we use the windowing approach to successively move up the image to determine the likely location of the lane markings.  In each row, the mean of the detected pixels provides the data for the polynomial curve fit.

- As shown below, the detected pixel means are fit with a quadratic polynomial (yellow lines).

  ![alt text][image24]

- Finally, these yellow lines are sampled to form the edges of a polygon, that is colored green, and then projected back onto the original road surface using the inverse perspective transformation as shown in the "Results" section to follow. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In lines 146 through 160 in `lanelines.py`, I computed the radius of curvature and offset from the center of the lane.

I computed the radius of curvature using the following analytical formula:

Given a parabola,

​	$x= Ay^2 + By + C$,

the radius of curvature $R(y)$ at any point y is given as:

​	$R(y;A,B,C) = \frac{(1+(2Ay+B)^2)^{3/2}}{|2A|}$

Before applying this result, we first have to realize that the coefficients we computed in the curve fit are in the _pixel_ space, and we need to map these to physical space (measured in meters). 

If $m_x$ is the number of meters per pixel in the $x$-direction, and $m_y$ is the number of meters per pixel in the $y$-direction, than we have the change of coordinates:

​	$x = X/m_x$, and $y = Y/m_y$.

Applying these change of coordinates, allows us to see that the parabolic fit in physical space is given by:

​	$X = \frac{m_x}{m_y^2}AY^2 + \frac{m_x}{m_y}By + m_xC$,

where we can read off the new fit coefficients as

​	$a =  \frac{m_x}{m_y^2}A$,

​	$b =  \frac{m_x}{m_y}B$,

​	$c = m_xC$.  (typo in the Udacity course for the C-coefficient.)

Then, we compute the radius of curvature (in meters) using $R(a,b,c)$.  

Computing radii of curvature for both lanes, we annotate the image with the result.

We also compute the lateral offset of the car in the lane (assuming the camera is located at the center of the front grille of the vehicle).

Here is an example of my result on a test image:

![alt text][image26]

## Results: Test Images##

Here are the results of applying the lane detection pipeline to the test images provided in the repository.

##![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]

## Results: Video

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](ann_project_video.mp4)

------

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The final pipeline works well and produces stable results.
- An important aspect of the development was to generate visualization tools to see the effects of the various thresholding parameters.  This was _very_ important.  Without the help of the visualization tools to see how the various thresholds were affecting the various contributions to the thresholding scheme, I was essentially a monkey at a typewriter.  
- I am concerned that the result shown here is the result of too much peeking at the results and tinkering with the parameters.  In other words, I have overfit the parameters for the given video, and it may be fragile to other scenarios.
- If I had more time (and money) to continue this project, I would want to integrate more robust approaches such as a filtering approach that retains a state for the lane fit coefficients, and then updates the coefficients using each successive image as a measurement.
- I would also want to test the approach on more diverse videos to verify robustness. 
- Other improvements include ensuring that the code does not break if no lane markings are detected.