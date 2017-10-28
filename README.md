# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/recovery_behavior.png "Flipped Image"
[image2]: ./img/center.png "Center Driving"
[image3]: ./img/recovery_behavior.png "Recovery Image"
[image4]: ./img/flip_c.png "Fliped Image 1"
[image5]: ./img/flip_l.png "Fliped Image 2"
[image6]: ./img/flip_r.png "Fliped Image 3"
[image7]: ./img/tran_center.png "Translated Image 1"
[image8]: ./img/tran_left.png   "Translated Image 2"
[image9]: ./img/tran_right.png  "Translated Image 3"
[image10]: ./img/both_c.png "Result Image 1"
[image11]: ./img/both_l.png "Result Image 2"
[image12]: ./img/both_r.png "Result Image 3"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 is containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it includes comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 19-51) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Also, I add cropping layer to remove redundant portion of the image.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers to reduce overfitting (model.py lines 26, 29, 32, 35, 38, 47). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 174-177). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (`model.py` line 51).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to make network not only can drive smoothing but also adjust itself back to the center of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to search existing network that successfully tackles the driving problem.

My first step was to use a convolution neural network model similar to the paper of NVIDIA (End to End Learning for Self-Driving Cars). I thought this model might be appropriate because the paper demonstrates outstanding performance on the real vehicle and test it on real environment. The system learns from example to detect the lane line of a road without the need of explicit labels
during training. Even without lane line, the car can stay in the middle of the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add dropout layer after convolutional layer to prevent it happened again.

Then I try the different hyperparameter such as epochs and batch size to minimize training loss and validation loss.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle might fall off the track and hit both sides of the bridge. To improve the driving behavior in these cases, I record more driving data with recovery behavior from both sides of the road. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 19-49) consisted of a convolution neural network with the following layers and dropout layer to prevent overfitting.

Here is a visualization of the architecture

![](https://i.imgur.com/FfKado7.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in the middle. These images show what a recovery looks like starting from the left side and right side :

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

#### 4. Date augmentation

##### Flipped images

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]
![alt text][image6]

##### Horizontal shifts

We shift the images horizontally to simulate the effect of the car being at different positions on the road and add an offset corresponding to the shift to the steering angle. 

After the collection process, I then preprocessed this data by adding the Cropping2D layer to remove redundant potion. It helps network more focus on road features instead of tree or sky.

![alt text][image7]
![alt text][image8]
![alt text][image9]

##### Preprocessing Result

After applying filpped and horizontal shift, following images are what it look like.

![alt text][image10]
![alt text][image11]
![alt text][image12]

---

I finally randomly shuffled the dataset and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or underfitting. The ideal number of epochs was 10 as evidenced by several trials. I also used an Adam optimizer so that manually training the learning rate wasn't necessary.
