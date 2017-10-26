
[//]: # (Image References)

[image1]: ./images/1.png "Visualization"
[image2]: ./images/2.png "Sample count"
[image3]: ./images/3.png "Sample count after adding samples"
[image4]: ./images/4.png "Normalization"
[image5]: ./images/5.png "Custom test images"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
 1. Submission Files: All files included have been pushed into separete repo. report.html has been created, writeup_template.md has been filled. The python notebook has the necessary changes 
 2. Dataset summary: The dataset summary such as number of training and test examples has been printed in the ipython notebook. 
 3. Exploratory Visualization: All the classes images have been plotted to give an idea of the dataser. Also, a graph showing number of images in each class has also been plotted for the training dataset
 4. Preprocessing, Model architecture, model training, solution approach : Explained below
 5. Acquiring New Images: New images were downloaded. The images are placesd in a folder named 'custom_test_signs'
 6. Performance on new images: I tested on 5 new images and 4 of them were classified correctly, which gives an accuracy of 80%. 
 7. Model Certainty - Softmax Probabilities: The softmax probabilities have been printed in the notebook

You're reading it! and here is a link to my [project code](https://github.com/vigneshu/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 47429
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
  

Here is an exploratory visualization of the dataset.  It iterates through each class and finds the first image in the dataset which matches the class and displays it


Next, I use the y_train to get the class count and plot a bar graph. You can see this in cell 4.


![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I chose not to change it to grayscale because I felt the color gives good information for the network to classify the sign. 
As a first step, I also created more data from the training datasset. The dataset does not have equal images for all classes, but rather skewed towards certain classes. This means that features for classes with less images aren't learn't and the model may be biased with the classes with lots of data. To overcome this, I took the classes with less images and randomly rotated them and added it to the training data again.

Finally, I normalised the image because it helps speed up learning and gives better results. 


![alt text][image4]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					                                  | 
|:---------------------:|:-------------------------------------------------------------------------------:| 
| Input         		| Input = 32x32x3                                                                 | 
| Convolution 3x3     	| Convolutional. Input = 32x32x3. Output = 28x28x6. 1x1 stride, 'VALID' padding   |
| RELU					|												                                  |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6. 2x2 stride. 2x2 k-size                            |
| Convolution 3x3	    |  Convolutional. Input = 32x32x3. Output = 28x28x6. 1x1 stride, 'VALID' padding  |     				| RELU					|												                                  |	
| Max pooling	      	| 2x2 stride,  Input = 10x10x16. Output = 5x5x16. 2x2 stride. 2x2 k-size	      |
| Fully connected		| Fully Connected. Input = 400. Output = 120. dropout = 0.3 (70% keep)            |
| RELU					|												                                  |	
| Fully connected		| Fully Connected. Input = 84. Output = 10. dropout = 0.3 (70% keep)              |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet. I added a dropout to the given Lenet model. The learning rate used was 0.001 and a batch size of 128. I used 10 EPOCHS and optimizer used is AdamOptimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


Training adn accuracy: cell 12

My final model results were:
* training set accuracy of 98.5
* validation set accuracy of  95.7
* test set accuracy of 95.3

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
THe first architecture was basically the ones provided by the instructors. I felt this would be a good starting point to get started with as LeNet is popularly used and gives good results. 
* What were some problems with the initial architecture?
The main problem was the lack of data for certain classes. To overcome this a few images were rotated and added bacak to improve training. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
As mentioned above, Adding a dropout helped improving the training. Dropout prevents overfitting. WIthout the dropout the training gave in good results but the validation accuracy was low. This hinted overfitting and hence I added a dropout layer.
* Which parameters were tuned? How were they adjusted and why?
I tried out various EPOCH, BATCH size and learning rate. Increasing the EPOCH led to overfitting and therefore I stuck with the parameters given. I did not tune the other parameters also since it gave the required accuracy ayways.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
LeNet architecture was used. 
* Why did you believe it would be relevant to the traffic sign application?

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The accuracy was above 93% in the three datasets which suggests the model works well. The model had never seen the validation and test data sets but still produced an excelklent efficiency which suggests a robust model.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image5]

The ahead only, turn right ahead and stop signs have backgrounds which is not there in the training data. This will be a problem while classifying. 
THe images were also not 32x32. Therefor this had to be resized. The jpeg compression along with resizing might make it difficult for the classifier

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| Turn right ahead		| Turn right ahead 								|
| 60km/hr				| End of all speed and passing limits			|
| Stop Sign	      		| Stop Sign 					 				|
| 30km/h     			| 30km/h             							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.3

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only     								| 
| 1.0     				| Turn right ahead								|
| 0.93					| 60km/hr										|
| 0.95	      			| Stop Sign 					 				|
| 0.84				    | 30km/h      				        			|

TopKV2(values=array([[  1.00000000e+00,   2.91263595e-13,   4.18282036e-20,
          1.33308244e-24,   8.63458119e-26],
       [  1.00000000e+00,   3.52463303e-09,   1.21572918e-13,
          4.05887005e-17,   3.52737400e-18],
       [  9.93152142e-01,   6.22407999e-03,   3.56298959e-04,
          1.72246961e-04,   6.01530992e-05],
       [  9.52518106e-01,   3.75414789e-02,   8.35273787e-03,
          1.35644840e-03,   8.85113695e-05],
       [  8.41915071e-01,   1.58074439e-01,   1.04279925e-05,
          1.17205479e-09,   6.24726920e-12]], dtype=float32), indices=array([[35, 36, 34, 33, 19],
       [33, 37, 35, 40, 34],
       [34, 36, 38,  3,  6],
       [29, 14, 24, 28, 22],
       [ 6,  1, 40,  2,  0]], dtype=int32))

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
