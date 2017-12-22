# Self-Driving Car Engineer Nanodegree Program
## Semantic Segmentation

[//]: # (Image References)
[image1]: ./output_images/FCN.png
[image2]: ./output_images/FCN-2.png
[image3]: ./output_images/Upsampling.png
[image4]: ./output_images/skip-layers.png
[image5]: ./output_images/loss-plot.png
[image6]: ./runs/um_000013.png
[image7]: ./runs/um_000011.png
[image8]: ./runs/um_000015.png

### Introduction
The goals for the project are 

* To label the pixels of a road in images using a Fully Convolutional Network (FCN). 
* Each pixel is labeled either as Road or non-roads. The same classification can be applied to other object such as other cars, bicycle etc.
* However, as per current project requirements, I have used to classify only Raod and Non-roads.
* Project Uses "Fully convolutional network (FCN)" developed at [Berkeley](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). 
* FCN uses convutional layers uses VGG net with Downsampling and Upsampling inside the network
* Each layer in the covnet is a three-dimensional array of size h x w x d where h and w are spatial dimensions and d is the feature or channel dimension.
* VGG Net has seven layers and a Fully connected layer.
![alt text][image1]
* Each layer is downsized h/4, h/8 and h/16 and fully connected layer at h/32.
![alt text][image2]
* FCN uses a concept of Upsampling and Skip layers to maintain the spatial dimension of the entire image. 
![alt text][image3]
![alt text][image4]
* Upsampling uses a Transposed Convolution with kernel sizes of 4, 4 and 16 with Strides of 2, 2 and 8. The code is present in the main.py at lines 65 to 100.
* Network was trained for 20 epochs


# Rubric Points
## On average, the model decreases loss over time..
* The logs are presents in the nohup.out
* The loss function is plotted below. [Log File](./nohup.out)
![alt text][image5]

## Does the project use reasonable hyperparameters?
* Following hyper parameters were used in the project.
	* keep_prob_stat = 0.8
	* learning_rate_stat = 1e-4
	* epoch = 20
	* batch-size = 1

## Does the project correctly label the road?
* Output Images with Pixel labels are present below.

![alt text][image6]
![alt text][image7]
![alt text][image8]

* All other images are found in the [run](./runs) directory. 


# References

* [Berkley Paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [Jon Long and Evan Shelhamer CVPR15 Caffe Tutorial](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf)
* [CS231N 2017 - Stanford - Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo&t=1103s&index=11&list=PLzUTmXVwsnXod6WNdg57Yc3zFx_f-RYsq)


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
