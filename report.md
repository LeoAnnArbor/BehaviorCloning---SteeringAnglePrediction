# Behavioral Cloning 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview and project goals
---
In this project, I will use deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle. General steps taken are as follows:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

---
### Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* Behavior cloning.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
* report.md summarizing the results

---
## Model Architecture and Training Strategy

### 1. Architecture 

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) - the diagram below is a depiction of the nVidia model architecture.

<img src="./images/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text - including converting from RGB to YUV color space, and 2x2 striding on the 5x5 convolutional layers. I used `RELU` as activation layersThe Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). The final layer (depicted as "output" in the diagram) is a fully-connected layer with a single neuron. 

To prevent overfitting, I added dropout layers, L2 regularization and adjust learning rate of Adam optimizer to 0.0001. These strategies did, indeed, result in less bouncing back and forth between the sides of the road, particularly on the test track where the model was most likely to overfit to the recovery data.

### 2. Loading and Preprocessing

In training mode, the simulator produces three images per frame while recording corresponding to left-, right-, and center-mounted cameras, each giving a different perspective of the track ahead. The simulator also produces a `csv` file which includes file paths for each of these images, along with the associated steering angle, throttle, brake, and speed for each frame. My algorithm loads the file paths for all three camera views for each frame, along with the angle (adjusted by +0.25 for the left frame and -0.15 for the right), into two numpy arrays

Images produced by the simulator in training mode are 320x160, and require preprocessing prior to being fed to the CNN since it expects input images to be of size 200x66. To achieve this, the bottom 20 pixels and the top 50 pixels are cropped from the image and it is then resized to 200x66. The color space is converted from RGB to YUV as suggested by [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Because `drive.py` uses the same CNN model to predict steering angles in real time, it requires the same image preprocessing.

### 3. Jitter

To minimize the model's tendency to overfit to the conditions of the test track, images are "jittered" before being fed to the CNN. The jittering consists of a randomized brightness adjustment, and a randomized horizon shift.The effects of the jitter can be observed in the sample below.

Original
<img src="./images/original.png?raw=true">

Random brightness
<img src="./images/randomise.png?raw=true">

Horizontal shift
<img src="./images/jitter.png?raw=true">

### 4. Implementing a Python Generator in Keras

When working with datasets that have a large quantities of image data, Keras python generators are a convenient way to load the dataset one batch at a time rather than loading it all at once. Aforementioned image processing pipeline is applied to each batch separately


## Conclusion and Discussion

This project - along with most every other exercise in machine learning, it would seem - very much reiterated that it really is *all about the data*. Making changes to the model rarely seemed to have quite the impact that a change to the fundamental makeup of the training data typically had. 

I could easily spend hours upon hours tuning the data and model to perform optimally on both tracks, but to manage my time effectively I chose to conclude my efforts as soon as the model performed satisfactorily on both tracks. I fully plan to revisit this project when time permits.

Therefore, the best way to improve the model is to collecting additional driving data. Udacity provides a dataset that can be used alone to produce a working model. However, students are encouraged (and let's admit, it's more fun) to collect our own. Particularly, Udacity encourages including "recovery" data while training. This means that data should be captured starting from the point of approaching the edge of the track (perhaps nearly missing a turn and almost driving off the track) and recording the process of steering the car back toward the center of the track to give the model a chance to learn recovery behavior. It's easy enough for experienced humans to drive the car reliably around the track, but if the model has never experienced being too close to the edge and then finds itself in just that situation it won't know how to react.
