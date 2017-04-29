**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

[//]: # (Image References)

[image1]: ./test_images/solidYellowCurve.jpg "solidYellowCurve"
[image2]: ./test_images/grayscale.png "Grayscale"
[image3]: ./test_images/gaussianblur.png "GaussianBlur"
[image4]: ./test_images/canny.png "Canny"
[image5]: ./test_images/masked.png "Masked"
[image6]: ./test_images/hough_lines.png "Houghlines"
[image7]: ./test_images/weighted_image.png "WeightedImage"
[image8]: ./test_images/processed_image.png "Processed Image"

[linkname1]( ./white.mp4 )

[linkname2]( ./yellow.mp4 )

---

### Reflection

The Lane detection transformation from Image to Video is described below.

First step was to create a function to could read an image and use it to convert video file. 
![alt text][image1]

The whole transformation process is described below.


* Convert input image into Grayscale. This removes the entire color so we can focus only on the white and black areas.

![alt text][image2]

* Use Compter Vision Libraries(cv2 libraries) create a Guasian Blur. The guasian blur mechanism is used to reduce the image noise.

![alt text][image4]

* After the Guausian blur image is created, the next step is create a canny edge image which outlines the edges in an image. The canny image creates a outline of the entire image.

![alt text][image3]

* Next step is to create a Lane Lines by color and masking all the areas. This is done by calling region_of_interest function which converts the input Guausian blurred image to a masked image (both color as well as regions).

![alt text][image5]

* Masked image is converted from line space to points in hough space. A line in image space will be a single pointin Hough space. Opencv HoughLinesP funtion is used to convert which takes the edges as input along with rho, theta and Threshold. The HoughLines return the lines. 

![alt text][image6]

* The image needs to be merged(superimposed) with the original image. This is achieved by calling addweight method passing the original and masked image.

![alt text][image7]

* The output is the image with the overlayed detecting the line segments.

* Drawlines function was modifed to add weight to get the thicker line.

![alt text][image8]

* Once the function is created, its verified for various image and a a helper fucntion is created.
* The helper function was applied for Videos.

### Video White Lane

![linkname1]

### Video Yellow Lane

![linkname2]


###2. Identify potential shortcomings with your current pipeline

The current pipeline assumes a brigter images and straight line roads. The pipeline might not work for curved lanes. The pipeline might fall short for blurry images, shadows and night drives. Its doesn't take into accout the weather conditions and 

###3. Suggest possible improvements to your pipeline

