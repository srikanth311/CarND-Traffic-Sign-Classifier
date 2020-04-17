# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[sri-image1]: ./traffic-classifier/output_images/label_train_data.png "Visualization Training data set"
[sri-image2]: ./traffic-classifier/output_images/label_test_data.png "Visualization Test data set"
[sri-image3]: ./traffic-classifier/output_images/label_validation_data.png "Visualization validation data set"
[sri-image4]: ./traffic-classifier/output_images/sample_images_train.png "Random 5 color images in each label class"
[sri-image5]: ./traffic-classifier/output_images/sample_images_train_gray.png "Random 5 gray images in each label class"
[sri-image6]: ./traffic-classifier/output_images/augmented_image_for_label_6.png "Warped Image for Lable 6"
[sri-image7]: ./traffic-classifier/output_images/label_augmented_train_data.png "Augmented Train Data"
[sri-image8]: ./traffic-classifier/output_images/new_images.png "New Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is : 34799
* The size of validation set is : 4410
* The size of test set is : 12630
* The shape of a traffic sign image is : (32, 32, 3)
* The number of unique classes/labels in the data set is : 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][sri-image1]
![alt text][sri-image2]
![alt text][sri-image3]
![alt text][sri-image4]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After doing some research online, I did the following steps to preprocess the color (RGB) images.

* Apply Histogram equalization
* Conversion to Grayscale images
* Min Max scaling
* Data augmentation
* One hot encoding of labels

**Apply histogram equalization**
To improve the contrast of the images, i have used histogram equalization. Specifically applied CLAHE(Contrast Limited Adaptive Histogram Equalization).
Followed the example found here :
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html

**Conversion to Grayscale images**
Converting to grayscale images gives better results compared to color images as color images are not a good differntiator in some cases.
Also, it will be much faster to process gray scale images vs color images as the system has to process through 1 channel vs 3 channels.

**Min Max Scaling**
Applied min mac scaling to get the values between 0 and 1 range.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][sri-image5]

**Data augmentation**

I decided to generate additional data because some of the label classes have few data points. This can be clearly seen from the above bar charts. This will not generate good results during predicting. 
To add more data to the the data set, I used the following technique.
From scikit library, used a transform parameters for a randamized distribution.  

Here is an example of an an augmented image:

![alt text][sri-image6]

The difference between the original data set and the augmented data set is the following ... 

**One Hot Encoded Labels**

I have appliede one hot encoding to all the labels.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image - (1 represents one channel)| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32    				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128    				|
| Flatten				| outputs 2048									|
| Fully Connected		| outputs 1024									|
| RELU		            |           									|
| Dropout		        | 0.5		        							|
| Fully Connected		| outputs 265									|
| RELU		            |           									|
| Dropout		        | 0.5		        							|
| (Output Layer) Fully Connected| outputs 43							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer as the cost function. I tried with differnet learning rates while training the models. Some of the learning rates i have tried are 0.1, 0.0001, 0.001. Out of all these learning rates, i got the best result with 0.001 as learning rate.

I trained my model with just CPU based Amazon EC2 instance. It took almost > ~40 minutes to complete the training. Then i trained the model on GPU based Amazon EC2 instance (g4dn.2xlarge) and it finished less than a minute to process this training step.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.24%
* validation set accuracy of 95.78% 
* test set accuracy of 92.96%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started with Lenet architecture that was explained in the tutorial. I chose this architecture because Lenet is very good for extracting text from the images and it is a simple architectture.

* What were some problems with the initial architecture?
When I tried with Lenet architecture, I was getting around ~91% of accuracy with the validation dataset.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I have used the below methods to adjust the Lenet (original) architecture.

a. Added one more convolution layer (3rd layer) in the architecure.

b. Changed the output size of each CNN layers. 

c. Downsized with maxpool layer

d. Added one more fully connected layer


* Which parameters were tuned? How were they adjusted and why?

As described in the tutorial, learning rate was an important tuning parameter. I tried with various values and i found 0.001 is giving best results.

Also, tried with different number of "epochs" values. Some of the values i have tried are: 25, 40, 25, 35. And i see after 20 iterations, i am not getting any good results. So used this value in my final model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.



Here are some of German traffic signs that I found on the web:

[![alt text][sri-image8]

The first image might be difficult to classify because ...

In my tests, It was unable to classify the "speed limit (30km/h) as a correct image. It was detecting that as 20km/h image. After running it couple of times, it was able to identify it correctly. It could be because of text characters were not clear enough to classify with in the red bold circle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|3 - Speed limit (60km/h)|3 - Speed limit (60km/h)| 
|11 - Right-of-way at the next intersection|11 - Right-of-way at the next intersection|
|1 - Speed limit (30km/h)|1 - Speed limit (30km/h)|
|12 - Priority road|12 - Priority road|
|38 - Keep right|38 - Keep right|
|34 - Turn left ahead|34 - Turn left ahead|
|18 - General caution|18 - General caution|
|25 - Road work|25 - Road work|


As mentioned, in the initial runs, it was able to find 7 out 8 images correctly. But after executing the prediction code on the same images couple of times, i see it is able to detect 8 out 8 images correctly with an accuracy of 100%.

From these results, it might get difficult to predict when the images are not clear and if they are more distorted.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 97th cell of the Ipython notebook.

For most of my images that i selected, the probabilty of prediction is close to 100%.

**Correct Sign : 25 - Road work**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 56.15%         			| 25 - Road work   									| 
| 19.83%     				| 20 - Dangerous curve to the right 				|
| 18.00%					| 30 - Beware of ice/snow											|

**Correct Sign : 38 - Keep right**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%         			| 38 - Keep right   									| 
| 0.00%     				| 2  - Speed limit (50km/h)  				|
| 0.00%					| 14 - Stop 											|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I was unable to complete this before the project due date. I will work on this and will fine tune my model later.


