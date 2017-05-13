# Vehicle Detection Project

The goals / steps of this project are the following:

* Download the labeled Car and Non images from Udacity links
* Rename the images
* Shuffle and Normalize the images
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-image1.png
[image2]: ./output_images/car-image2.png
[image3]: ./output_images/car-image3.png
[image4]: ./output_images/car-image5.png
[image5]: ./output_images/car-hog-side-by-side1.png
[image6]: ./output_images/car-hog-side-by-side2.png
[image7]: ./output_images/car-hog-side-by-side3.png
[image8]: ./output_images/car-hog-side-by-side4.png
[image9]: ./output_images/box-mulitple4.png
[image10]: ./output_images/box.png
[image11]: ./output_images/sliding-window.png
[image12]: ./output_images/heatmap.png
[image13]: ./output_images/heatmap2.png
[image14]: ./output_images/non-car-image1.png
[image15]: ./output_images/non-car-image4.png
[image16]: ./output_images/heatmap3.png
[image17]: ./output_images/heatmap4.png
[image18]: ./output_images/heatmap5.png
[image19]: ./output_images/heatmap6.png
[image20]: ./output_images/false-positive.png
[video1]: ./output_final_project_Final_ver2.mp4




I have created following classes for completion of the project. The main python notebook depicts the execution of the code in the following order

1. Read the images from the directory and rename/store the images.
2. Display the car and non car images.
3. Display the hog Visualization of the car and non-car images 
4. Shuffle the car and non-car images randomly.
5. Preprossesing - Use Feature Extraction, Normalize, Create Training and test set.
6. Train Support Vector Machine Classifier and calculate the score
7. Store the values in pickle file.
8. Apply the Classifier to set of images, create a heat map to remove mutliple windows and False positives
9. Run the classifier on the pipeline video

### Classes Used in the project ###
* project-p5-final-Submission - Main iPython Notebook for the execution
* utils/ApplyClassifier.py - The class has method for creating pipeline for single image, video frmaes
* utils/FeatureExtractorUtil.py - This class has methods for finding features in images
    * get_hog_features - HOG Feature Extraction
    * bin_spatial - Function to compute binned color features 
    * color_hist - Function to compute binned color features
    * extract_features - function to extract features from a list of images
    * slide_window - Sliding Window Creation
    * draw_boxes - Function to draw bounding boxes
    
* utils/Smoothening.py - The class has methods for heat Mat Creation
* utils/Classifier.py - This class has methods for Standardization and Support Vectory classifier training and score

###Histogram of Oriented Gradients (HOG)###

##### Car Images #####
![alt text][image1]
![alt text][image2]

##### Non Car Images #####
![alt text][image14]
![alt text][image15]

The first step in image classification is to simplify the image by extracting the important information contained in the image and leaving out the rest. A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information. 

I then explored different color spaces and different `get_hog_features.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like for Cars.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image5]
![alt text][image6]

####2. Explain how you settled on your final choice of HOG parameters.

HOG feature extraction  converts an image of fixed size to a feature vector of fixed size. I used below configuration for Hog Feature Extraction.

* color_space='YCrCb' - Tried various color spaces such as 'RGB', HSV, LUV, HLS, YUV. 
* spatial_size=(32, 32) -
* hist_bins=32
* orient=9
* pix_per_cell=8
* cell_per_block=2
* hog_channel='ALL'

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using SVC. The `Classifier.SVC_Normalize` uses the SVC and train the classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this. The following code has the details.

 `FeatureExtractionUtil.slide_window`

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
![alt text][image9]
![alt text][image10]
![alt text][image12]
![alt text][image13]

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

Here are some example images:
![alt text][image10]

### Video Implementation

Here's a [link to my video result](./output_final_project_Final_ver2.mp4)
Youtube Link [Final Display Video](https://www.youtube.com/watch?v=UmqTIQ1BcnY)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image10]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipleine had many False poitives with Advanced Lane Detection Video Pipeline. Adding more data to the training and test set and improving the sliding window function could reduce the false positives.

![alt text][image20].

I reffered following links for more details.
https://www.learnopencv.com/image-recognition-and-object-detection-part1/

https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906


