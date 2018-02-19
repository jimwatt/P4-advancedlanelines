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


## Authors

* **James Watt**

<!--## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details-->

## Acknowledgments
This project is a submission to the Udacity Self-Driving Car nanodegree:

* https://github.com/udacity/CarND-Advanced-Lane-Lines.git

