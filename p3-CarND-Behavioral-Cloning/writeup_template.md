**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "Nividia Architecture"
[image2]: ./examples/resized-Image.png "Resized Image"

---
My project includes the following files:
* model.py containing the script to create and train the model  
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results




####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                    | Source code for reading, data augmentation, prepross, data augmentation and Network.|
| `drive.py`                   | Implements model architecture and runs the training pipeline.                      |
| `Model.h5`                 | JSON file containing model architecture in a format Keras understands.             |
| `drive.py`                   | Model weights.                                                                     |
| `video_output` | [Youtube Link](https://youtu.be/xakm7k-E9K0) |
| `drive.py`                   | Implements driving simulator callbacks, essentially communicates with the driving simulator app providing model predictions based on real-time data simulator app is sending. |



####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses [NVidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) Architecture with modifications. 

Below is the details on the architecture which I used.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 66x200x3 RGB image   							| 
| Normalization         		| 							| 
| Normalization         		| 	((20,0), (0,0))						| 
| Convolution 5x5     	| 2x2 stride, valid padding 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding 	|
| RELU					|												|
| Convolution 3x3	    | 2x2 stride, valid padding  |
| RELU					|												|
| Flatten	      	| Flatten. Output = 100. 				|
| Fully connected		| Input = 100. Output = 50.  |
| RELU					|												|
| Fully connected		| Input = 50. Output = 10.  |
| RELU					|												|
| Dropouts					|												|
| Fully connected		| Input = 10. Output = 1.  |



####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

####4. Appropriate training data

I used the simualator [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity for training the network.

I used the simualator [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity for training the network.

The Simuator dataset consists of images and a CSV file. Total of 24108 images with size of each image (160,320,3) captured from the 3 cameras mounted on center, left and right of the car. The dataset also contains a CSV file which has metadata about the images.

The metadata consists of Image paths, Steering angle, throttle, break and speed data at various interwals during the training.

The data can be used to directly train the network. However, the data needs augmentation for various scneraios such as sharp turns, shadows, tress etc. 

I used the python CSV functions to read the training data from CSV. I didn't clean up the data and used only resizing. Initially I used CV2 image read function. However, I had difficulties using the simulator. Later I used the approach as suggested in the [link](https://medium.com/@yazeedalrubyli/behavioral-cloning-tiny-mistake-cost-me-15-days-23dd13a3b525)


###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with fit_generator. It was too slow and took hours to train. Rather I used the tradditional approach to store the data in memory which was much faster and more easy to test.

My first step was to use a convolution neural network model similar to the Nivida. I chose the model based as a starting point. However, the model is comlicated and I had to reduce the complexity.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with 



To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| Flatten. Input = 5x5x16. Output = 400. 				|
| Fully connected		| Input = 400. Output = 120.  |
| RELU					|												|
| Dropouts					|												|
| Fully connected		| Input = 120. Output = 84.  |
| RELU					|												|
| Dropouts					|												|
| Fully connected		| Input = 84. Output = 43.  |
| Softmax				|        									|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
