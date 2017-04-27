# **Traffic Sign Recognition** 

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

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Reen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the frequency of the
different traffic sign classes.

![histogram1]

Another way to inspect the data is to look at different samples from each class:

![overview]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the signs are
clearly distinguishable without the color information.
If a human can distinguish the signs in grayscale, a neural network should also do.

Here is an example of a traffic sign image before and after grayscaling.

![grayscale]

As a next step, I normalized the image data using `tf.image.per_image_standardization`.
This way, different input vectors have the same influence on the corrections caused by the gradient error vectors.

I decided to generate additional data because more data is always better ;-). 
To add more data to the data set, I flipped some image classes either horizontally, vertically or both.
For some classes flippig leads to a change in meaning, for instance, flipping a "keep right" sign 
horizontally leads to a "keep left" sign.

Here is an example of an original image and an augmented image:

![flip]

The augmentation resulted in the following label frequencies:

![histogram2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   	    			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU          		| etc.        									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16      				|
| Flatten				|           									|
| Fully connected		| 120 outputs									|
| RELU          		| etc.        									|
| Dropout				| 50%											|
| Fully connected		| 84 outputs									|
| RELU          		| etc.        									|
| Dropout				| 50%											|
| Logits                | 43 outputs                                    |
| Softmax               |                                               |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a batch size of 128 and 40 epochs.
The learning rate has been set to 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My development took the following evolution:
1. Baseline:
  * Copied LeNet for grayscale images from LeNet example
  * Converted input images to grayscale
  * Applied tensorflow `per_image_standardization`, i.e. so that images have zero mean and unit norm
  * Parameters: rate = 0.001 EPOCHS = 20 BATCH_SIZE = 128
  * Best validation acc.: 0.933
2. Dropout Layer after fully connected layer 1
  * Parameters: rate = 0.001 EPOCHS = 40 BATCH_SIZE = 128
  * Best validation acc.: 0.968 (epoch 38)
3. Dropout Layer after fully connected layer 2
  * Parameters: rate = 0.001 EPOCHS = 40 BATCH_SIZE = 128
  * Best validation acc.: 0.967 (epoch 27)
4. Like 2. and L2 regularization of the weights
  * Parameters: rate = 0.001 EPOCHS = 40 BATCH_SIZE = 128 beta = 0.01
  * Best validation acc.: 0.951 (epoch 29)
5. Like 2. and augmentation of data by flipping some classes of images
  * Parameters: rate = 0.001 EPOCHS = 40 BATCH_SIZE = 128
  * Best validation acc.: 0.973 (epoch 33)
6. Reimplemented LeNet in tf.slim applying two dropout layers and no regularization
  * Parameters: rate = 0.001 EPOCHS = 40 BATCH_SIZE = 128
  * Best validation acc.: 0.972
  * Test accuracy: 0.948

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.971 
* test set accuracy of 0.948

I chose to base my development on LeNet from the previous lessons.
Distinguishing different handwritten letters from one another is, in my opinion,
not much different then differentiating different traffic signs.
Using the simple LeNet together with grayscale and normalized images
already lead to a validation accuracy of 0.933.
I attributed the sub-optimal performance to overfitting.
Using this as a baseline, I experimented with a dropout layer applied
after one of the two fully connected layers to overcome overfitting.
This led to an improved validation accuracy of ~0.967.
Adding L2 regularization decreased the performance to ~0.951.
Thereafter, I experimented with several major changes to the network,
like increasing the size of the fully connected layers or the depth of the convolutional filters,
which all led to underfitting.
Then I chose to augment the dataset as described above. This led to an improvement
of the validation accuracy to 0.973.
Further, I rewrote the network using `tf.slim`, which vastly improved the readability of the code.
In this implementation, I chose to apply two dropout layers, one after each fully connected layer,
reaching the same accuracy as before.
The accuracy of the final model gives hint, that there might be room for improvement.
The training accuarcy of 0.999 indicates, that further training using the same data will not improve
the validation and test accuarcies of 0.971 and 0.948 further.
Overall, these values indicate, that the model works well within its abilities.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I photographed in my hometown:

![mysigns]

The first image might be difficult to classify because it has some grafitti on it.
The second and third signs were hanging directly besides traffic lights, which glowed into the signs.
The other three signs are in good condition.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| Priority road			| Priority road									|
| Stop                  | Stop                                          |
| Ahead only	      	| Ahead only					 				|
| Yield					| Yield											|
| Yield					| Yield											|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.
This compares favorably to the accuracy on the test set of 0.948.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the six images, the model is very sure with its prediction. For the correct class, the softmax probability came out to be ~1.0, with the probabilities
of the next 4 classes being at least below 10^-8.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


[histogram1]: histogram.png "Frequency of the different traffic sign classes"
[overview]: overview.png "Overview over the classes, showing 10 samples of each class"
[grayscale]: grayscale.png "Example image shown in color and after conversion to gray scale"
[histogram2]: histogram2.png "Frequency of the different traffic sign classes after augmentation"
[flip]: flip.png "Example showing horizontal flippig of a keep right sign turning into a keep left sign"
[mysigns]: traffic_signs_chemnitz.png "Six traffic signs I photographed in my hometown"