## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
<<<<<<< HEAD


#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./images/traffic-images-Label-38-before-normalization.png "Before Normalization"
[image10]: ./images/traffic-images-Label-38-after-normalization.png "After Normalization"
[image11]: ./images/DataSet-Histo-Before-Normalization.png "Dataset Normalization"
[image12]: ./images/DataSet-Histo-After-Normalization.png "Dataset Normalization"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/manjunathc/udacity-driveless-car-program/blob/master/p2-CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier-final.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I downloaded the dataset from below link [Traffic Data Set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). 
The dataset contained 
* Training set
* Test Set
* Validation Set

I used the pickle library to load the data set

* The size of training set is 34799 with 34799 Labels
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a Histo chart showing how the data before Normalization. The chart depicts the Unique classes and the number of images assocaited with each unique Label. 

![Dataset Before Normalization][image11]

![Dataset After Normalization][image12]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to process the images without normalization and plot the images for all classess. For simplicity I decided to display only first 10 images per class. I was visually able to see many of the images were unclear with blurring. 

I used OpenCV Normalization function to normalization every image in the dataset. 

Here is an example of a traffic sign image before and after Normalization.

Before Normalization

![Label 38 before Normalization][image9]

After Normalization

![Label 38 before Normalization][image10]

I haven't used to generate additional data as the testset was ~30% of the original imageset. The validation set was ~13%.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16  |
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


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used 20 Epochs and batch size of 128 with AdamOptimizer. Other Hyperparameter with learning rate of 0.001. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

Initalally I trained the network without adding the dropouts and Max Pooling with the same Epoch, Optimizer and batch Size. The accuracy was less than 85%.
Later I decided to add the Max Pooling and Dropouts. The accuracy remained around 89% with the validation set. I increased the Epochs to 150. That helped the validation accuracy around 93%.
Later, I added normalization to the images which improved the accuracy and efficiency. With just 20 epochs I was able to see an improvement of validation accuracy more than 93%. 
With more epcohs the accuracy could be improved much better.

My final model results were:
* training set accuracy of ? 99.6 % 
* validation set accuracy of ? 95.4 % 
* test set accuracy of ? 93.5 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet-5 is the latest convolutional network designed for handwritten and machine-printed character recognition. I started with Lenet's architecture with  Convolution 3x3. The output was 5, 5, 3, 6. without Image normalization, Max Pooling or Dropouts. 
* What were some problems with the initial architecture?
The validation and testing data accuracy was less than 85-90% even though the training accuracy was around 90%. I had to train for the longer duration with many iterations (Epochs > 150). It was time consuming and not accurate.

* How was the architecture adjusted and why was it adjusted? 
I added additional layer with output of 5, 5, 3, 12 with the initial Layer. Additional to that I aaded Max pooling along with dropout at multiple layer using an activation function. This was used to reduce overfitting of the data and to improve the accuracy. Apart from that I did normalization of entire training set to improve the efficiency of training data.


* Which parameters were tuned? How were they adjusted and why?
Initially I started adding Dropouts followed by Max Pooling. I added Training dataset Normalization to improve overall efficiency.


* What are some of the important design choices and why were they chosen? 
Adding the Max Pooling and Dropout helps in overfitting with AdamOptimizer.

If a well known architecture was chosen:
* What architecture was chosen?
Lenet's architecture was chosen based on the Latest avaialble architecture for Character recognition.
* Why did you believe it would be relevant to the traffic sign application?
All Traffic data contains symbols. Lenet's model works well with Symbols
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The accuracy of the training, validation proved with 20 iterations that the model works well with Traffic Images.  

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


||||||| merged common ancestors
=======


#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./images/traffic-images-Label-38-before-normalization.png "Before Normalization"
[image10]: ./images/traffic-images-Label-38-after-normalization.png "After Normalization"
[image11]: ./images/DataSet-Histo-Before-Normalization.png "Dataset Normalization"
[image12]: ./images/DataSet-Histo-After-Normalization.png "Dataset Normalization"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/manjunathc/udacity-driveless-car-program/blob/master/p2-CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier-final.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I downloaded the dataset from below link [Traffic Data Set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). 
The dataset contained 
* Training set
* Test Set
* Validation Set

I used the pickle library to load the data set

* The size of training set is 34799 with 34799 Labels
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a Histo chart showing how the data before Normalization. The chart depicts the Unique classes and the number of images assocaited with each unique Label. 

![Dataset Before Normalization][image11]

![Dataset After Normalization][image12]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to process the images without normalization and plot the images for all classess. For simplicity I decided to display only first 10 images per class. I was visually able to see many of the images were unclear with blurring. 

I used OpenCV Normalization function to normalization every image in the dataset. 

Here is an example of a traffic sign image before and after Normalization.

Before Normalization

![Label 38 before Normalization][image9]

After Normalization

![Label 38 before Normalization][image10]

I haven't used to generate additional data as the testset was ~30% of the original imageset. The validation set was ~13%.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16  |
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


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used 20 Epochs and batch size of 128 with AdamOptimizer. Other Hyperparameter with learning rate of 0.001. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Initalally I trained the network without adding the dropouts and Max Pooling with the same Epoch, Optimizer and batch Size. The accuracy was less than 85%.

Later I decided to add the Max Pooling and Dropouts. The accuracy remained around 89% with the validation set. I increased the Epochs to 150. That helped the validation accuracy around 93%.

Later, I added normalization to the images which improved the accuracy and efficiency. With just 20 epochs I was able to see an improvement of validation accuracy more than 93%. 

With more epcohs the accuracy could be improved much better.


My final model results were:
* training set accuracy of ? 99.6 % 
* validation set accuracy of ? 95.4 % 
* test set accuracy of ? 93.5 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet-5 is the latest convolutional network designed for handwritten and machine-printed character recognition. I started with Lenet's architecture with  Convolution 3x3. The output was 5, 5, 3, 6. without Image normalization, Max Pooling or Dropouts. 
* What were some problems with the initial architecture?
The validation and testing data accuracy was less than 85-90% even though the training accuracy was around 90%. I had to train for the longer duration with many iterations (Epochs > 150). It was time consuming and not accurate.
* How was the architecture adjusted and why was it adjusted? 
I added additional layer with output of 5, 5, 3, 12 with the initial Layer. Additional to that I aaded Max pooling along with dropout at multiple layer using an activation function. This was used to reduce overfitting of the data and to improve the accuracy. Apart from that I did normalization of entire training set to improve the efficiency of training data.

* Which parameters were tuned? How were they adjusted and why?
Initially I started adding Dropouts followed by Max Pooling. I added Training dataset Normalization to improve overall efficiency.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Adding the Max Pooling and Dropout helps in overfitting with AdamOptimizer.

If a well known architecture was chosen:
* What architecture was chosen?
Lenet's architecture was chosen based on the Latest avaialble architecture for Character recognition.

* Why did you believe it would be relevant to the traffic sign application?
All Traffic data contains symbols. So the model works well with Symbols
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The accuracy of the training, validation proved with 20 iterations that the model works well with Traffic Images. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
