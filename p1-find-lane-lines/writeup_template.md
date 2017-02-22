#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./converted_image.png "Grayscale"

[video1]: ./white.mp4

[video2]: ./yellow.mp4

---

### Reflection

My pipeline consisted of 5 steps. 

My first step was to create a function to could read an image and convert into an transformed image. The whole transformation contained following process.

* Convert input image into Grayscale. This removes the entire color so we can focus only on the white and black areas. 
* Use Compter Vision Libraries(cv2 libraries) create a Guasian Blur. The guasian blur mechanism is used to reduce the image noise.
* After the Guausian blur image is created, the next step is create a canny edge image which outlines the edges in an image. The canny image creates a outline of the entire image.
* Next step is to create a Lane Lines by color and masking all the areas. This is done by calling region_of_interest function which converts the input Guausian blurred image to a masked image (both color as well as regions).
* Masked image is converted from line space to points in hough space. A line in image space will be a single pointin Hough space. Opencv HoughLinesP funtion is used to convert which takes the edges as input along with rho, theta and Threshold. The HoughLines return the lines. 
* The image needs to be merged(superimposed) with the original image. This is achieved by calling addweight method passing the original and masked image.
* The output is the image with the overlayed detecting the line segments.
* Drawlines function was modifed to add weight to get the thicker line.

![alt text][image1]

* Once the function is created, its verified for various image and a a helper fucntion is created.
* The helper function was applied for Videos.

[video1]: ./white.mp4

[video2]: ./yellow.mp4


###2. Identify potential shortcomings with your current pipeline

The Masked Image  parameters 

###3. Suggest possible improvements to your pipeline

Improve Hough 
