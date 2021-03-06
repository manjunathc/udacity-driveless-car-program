# Advanced Lane Finding Project #

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboardcorner-image1.png "Chessboard Image1"
[image2]: ./output_images/chessboardcorner-image2.png "Chessboard Image2"
[image3]: ./output_images/chessboardcorner-image3.png "Chessboard Image3"
[image4]: ./output_images/original_distorted.png "Original and Undistorted Image - SideBySide"
[image5]: ./output_images/original_undistorted.png "Original Undistorted - After Calibration"
[image6]: ./output_images/warped.png "Warped (Birds View)"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./output_images/sliding_window.png "Sliding Window"
[image9]: ./output_images/sliding_window2.png "Sliding Window - Subsequent frames"
[image10]: ./output_images/final_display.png "Final Display Image"
[image11]: ./chess_calibration_images/calibration3.jpg "Non Calibrated Images"
[image12]: ./chess_calibration_images/test_images/test1.jpg "Test Image for Visualization"
[image13]: ./chess_calibration_images/test_images/test1.jpg "Another Image for Visualization"
[video1]: ./output_final_project.mp4 "Project Video"
[video2]: ./output_final_project_challenge.mp4 "Challenge Video"
[image14]: ./output_images/original_distorted2.png "Original and Undistorted Image - SideBySide"
[image15]: ./output_images/original-threshold-side-side.png "Undistorted and Binary Warped Image - SideBySide"
[image16]: ./output_images/original-threshold-side-side2.png "Undistorted and Binary Warped Image - SideBySide"
[image17]: ./output_images/original-threshold-side-side3.png "Undistorted and Binary Warped Image - SideBySide"
[image18]: ./output_images/original-threshold-side-side4.png "Undistorted and Binary Warped Image - SideBySide"
[image19]: ./output_images/original-threshold-side-side5.png "Undistorted and Binary Warped Image - SideBySide"
[image20]: ./output_images/original-threshold-side-side6.png "Undistorted and Binary Warped Image - SideBySide"
[image21]: ./output_images/original-threshold-side-side7.png "Undistorted and Binary Warped Image - SideBySide"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I have created following classes for completion of the project. The main python notebook depicts the execution of the code in the following order

1. Calibrate the images using Chessboard corners and save the object points/Images points
2. Use test images for testing Undistortion, threshold, Binarywarped, histogram, Sliding Window and Display images
3. Verify the process for various images and calibrate accordingly
4. Create a pipeline for Images/Vides
5. Create a video using the pipeline
6. Test the pipeline for various Challenging videos.
### Classes Used in the project ###
* Final_project_submission - Main iPython Notebook for the execution
* CalibrationImages.py - The class has methods for identifying Chessboard Image corners, camera calibration, undistort and Binary Warped Images 
* Threshold.py - This class has methods for Threshold Calculation. 
    * Sobel Threshold
    * Magnitude Threshold
    * Direction of Gradient
    * Combined Threshold
* PlotImages.py - The class has methods for calculation and plotting of histogram, warped Image, Sliding window and Final Display
* PlotImgagesVideo.py - This class generates has the pipeline for final video generation
* Line.py - This class saves intermediate line parameters, calculating curvature and offset.



## Pipeline (single images)

Below are the set of images that depicts the various pipeline


### 1. Provide an example of a distortion-corrected image.

#### Chessboard Image ####
![alt text][image11]

#### Chessboard with Corners ####
![alt text][image1]
![alt text][image2]
![alt text][image3]

#### Distorted - Undistorted Images. Side-By-Side ####
![alt text][image14]
![alt text][image4]

#### Histogram ####
![alt text][image7]

#### Sliding Window. Side-By-Side ####
![alt text][image8]
![alt text][image9]

#### Final Display Image - With Curvature and offset ####
![alt text][image10]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 67 through 93 in `Threshold.py`).  Here's an example of my output for this step.

### Undistorted - Binary Warped Images. Side-By-Side ###
![alt text][image15]
![alt text][image16]
![alt text][image17]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `getBinaryWarped()`, which appears in lines 60 through 76 in the file `CalibrationImages.py` (imageUtils/CalibrationImages.py).  The `getBinaryWarped()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points for a sample Image:
```python
SRC
[[  585.           460.        ]
 [  203.33332825   720.        ]
 [ 1126.66662598   720.        ]
 [  695.           460.        ]]
DST
[[ 320.    0.]
 [ 320.  720.]
 [ 960.  720.]
 [ 960.    0.]]
 ```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image8]
![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented these step in lines 39 through 49 in my code in `imageutils/Line.py` in the function `ln_get_curvature()`, `ln_get_offset_from_center()`  for calulating curvature and offset respectively. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

 Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a Project [video1](./output_final_project.mp4)
Youtube Link [Final Display Video](https://www.youtube.com/watch?v=HmuTQVL8IUc)

Here's a Challenging [video2](./output_final_project_challenge.mp4)
Youtube Link [Challenge Video](https://www.youtube.com/watch?v=-pgAdfJtLdk)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Mainly I used all techniques as per the Udacity Videos. However, I had issues especially with the threshold calulation. The videos has very limited inputs on calculation of threshold. I referred few github links and following link to get more prespective.
https://chatbotslife.com/advanced-lane-line-project-7635ddca1960

Even with these , the challenging videos had issues. Appreciate some help around these

