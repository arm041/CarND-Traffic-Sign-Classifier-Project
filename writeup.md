# ** Traffic Sign Recognition** 
---

** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/test.png "Visualization"
[image2]: ./examples/validation.png "Visualization2"
[image3]: ./examples/training.png "Visualization3"
[image4]: ./examples/preprocess.png "Preprocess"
[image5]: ./examples/Rotated.png "Rotated"
[image6]: ./examples/1.png "Traffic Sign 1"
[image7]: ./examples/2.png "Traffic Sign 2"
[image8]: ./examples/3.png "Traffic Sign 3"
[image9]: ./examples/4.png "Traffic Sign 4"
[image10]: ./examples/5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
The link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas and the numpy libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 which means the height and width of the image are each 32 pixels and there are 3 channels for the color
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. There are 3 bar charts for the training, validation, and test set showing how many numbers of each class of the traffic signs are included in them. Also there is a pie chart that shows roughly the percentage of these classes such that they can be compared to each other. 

![alt text][image1]
![alt text][image2]
![alt text][image3]


###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because this can help to reduce the algorithms focus to learn based on color and hence make it more robust when it sees the same sign with different background color or maybe even the sign in another color. 

Also I decided to add some elements to the training set. For this I decided to augment the training set with a rotated version of all the images. So with this I doubled the size of my training set but got different versions of the traffic sign such that the classifier could generalize on the training set and give better results on unseen data. 
For rotating the images, I did it with a random variable that decides whether to rotate the image clockwise or counter clockwise for 5 degrees such that the sign are a little to the left or right and classifier can learn better the whole sign even if the image is a little rotated or the perspective is not completely from up front but a little tilted to a side. 

As a last step, I normalized the image data for the training, validation and test set also such that the grayscaled image became more smooth and the sign could be classified better. 
Here is an example of a traffic sign image before and after preprocessing:

![alt text][image4]


Here is an example of an augmented image:

![alt text][image5]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               | input: 5x5x16 outputs 400
| Fully connected		| matmul operation in tensorflow output size 120 |
| RELU					|												|
| Drop Out layer		| keep probability determined at training		|
| Fully connected       | matmul operation in tensorflow output size 84  |
| RELU                  |                                               |
| Fully connected       | matmul operation in tensorflow output size 43  |
 
To train the model, I used an adam optimizer so I didn't tune all the parameters by hand. The parameters I chose where the Epochs wich I decided to be 20 by experimenting. The learning rate used was 0.0008 and also a batch size of 128 was considered good for the network training.  

My final model results were:

* training set accuracy of 99.6%
* validation set accuracy of 94.8%
* Test set accuracy of 92%

For this project I chose the LeNet architecture because it is one of the most famous and useful architectures used for image classification. This architecture works on 32x32 images and this also helped me to be able to use it directly with the data provided to me. 
At first the architecture was clearly overfitting to the training set data and hence I had to also make a change to the architecture. The change I made was to introduce a dropout layer. This drop out layer is introduced in the fully connected part of the architecture as described above in the table where all the layers of the architecture are written. 
After obtaining the result it seems clear that the architecture has done a very good job on the data. The accuracy on the training set is immensely high, which might still be an indicator of overfitting, but the results on the validation set and the test set, as well as the 5 extra images found on the web, show that this model architecture has worked pretty well. 

The code for the architecture and the hyper parameters are written in the 5th code snippet of the notebook. The training and validation of the model is then done in the 6th code snippet, where the model is trained and validated through the 20 epochs. After that in the 7th code snippet the model was checked and used on the test set. 

After these steps to test the model on some new images I found 5 new German traffic signs in the INternet. Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because the triangle is not perfect and the sign is a little rotated to the right also the sign itself is not that clearly depicted in the image. 
The second image might be difficult to detect because it is rotated and tilted to the left and also there is another sign, from the back just behind the sign that adds another circle to the image. This might be a little difficult for the model to detect which one is the actual sign that has to be classified. 
The third image is a little difficult to detect because there is also a red car in the image, that could kind of look like it is part of the sign itself. 
For the fourth image it can be said that there are a lot of colors and noises in the image. the sign itself is clear and the perspective is from up front. 
The fifth image is completely unclear and the lighting situation is very bad for the sign. This will probably make it difficult to detect, even for a human eye that is a little tired. So this one would also be difficult to detect for the network in my opinion. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| No Entry     			| No Entry 										|
| Road narrows on the right| Road narrows on the right					|
| General caution      		| General caution					 		|
| Vehicle over 3.5 metric tons prohibited			|Vehicle over 3.5 metric tons prohibited      							|

The model was able to correctly guess all 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%, because the test set consisted of much more images and it is possible that the classifier make some mistake on them. 

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work sign(probability of almost 100%), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|        almost 1 	            | 						Road work			    | 
|       6.80e-30				|  	        			Pedestrians						|
| 			5.74e-35       		| 		          Dangerous curve to the right								|
| 	     5.55e-38 		     	| 				Right-of-way at the next intersection             	 				|
| 		almost 0		        |      			    Speed limit (20km/h)            				|


For the second image the no entry sign was classified correctly and the top candidates were as follows: 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|      7.11e-01                 |             No entry                                  | 
|      2.84e-01                 |                 Turn left ahead                              |
|        2.89e-03               |                    Stop                           |
|      4.17e-04                 |                Keep right                               |
|        2.59e-05               |               Yield                                |

For the third image again the model was pretty sure about the winner of the classification and all the other candidates had small chances of being selected. 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|     9.92e-01                  |  Road narrows on the right                                             | 
|      7.24e-03                 |    Pedestrians                                           |
|     1.65e-05                  |             Traffic signals                                  |
|    8.52e-08                   |           Children crossing                                    |
|      1.79e-08                 |             Right-of-way at the next intersection                                  |

For the fourth image the general caution sign was predicted correctly with the following softmax values:  

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|    almost 1                   |     General caution                                          | 
|     1.56e-08                  |     Traffic signals                                          |
|      1.72e-14                 |            Pedestrians                                   |
|     1.47e-17                  |             Right-of-way at the next intersection                                  |
|    9.30e-23                   |                Go straight or left                               |

For the fifth sign it was again almost completely decided without doubt by the model. Below the probabilities can be seen: 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
|    9.99e-01                   |  Vehicles over 3.5 metric tons prohibited                                             | 
|      2.10e-04                 |      End of no passing                                         |
|      5.15e-05                 |         End of all speed and passing limits                                      |
|     1.18e-05                  |          No passing                                     |
|      1.96e-06                 |        Roundabout mandatory                                       |



