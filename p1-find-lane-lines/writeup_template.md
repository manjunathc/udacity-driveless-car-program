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

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

My first step was to create a function to could read an image and convert into an transformed image. The whole transformation contained following process.

* Convert input image into Grayscale. This removes the entire color so we can focus only on the white and black areas. 
* Use Compter Vision Libraries(cv2 libraries) create a Guasian Blur. The guasian blur mechanism is used to reduce the image noise.
* After the Guausian blur image is created, the next step is create a canny edge image which outlines the edges in an image. The canny image creates a outline.
* 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline

The Masked Image masked 

###3. Suggest possible improvements to your pipeline

Improve Hough 
