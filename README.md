# Behavioral Cloning

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

