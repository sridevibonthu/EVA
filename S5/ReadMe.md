# EVA - Session 5



## Team Members
1. B Sridevi  - sridevi.b@vishnu.edu.in
2. Abhinav Dayal - abhinav.dayal@vishnu.edu.in
3. A. Lakshmana Rao - 18pa1a0511@vishnu.edu.in
3. Sanjay Varma G - 18pa1a1211@vishnu.edu.in

## First Model 
[File Link](https://github.com/sridevibonthu/EVA/blob/master/S5/EVA4_S5_01.ipynb)

### Observation
* The input is mono and has very simple features. Probably we need not go beyond 16 channels.
* We can do initial max pooling after we reach receptive field of 5x5
### Target
* Less than 20000 parameters
* Less than 15 epochs
### Results
* Number of Parameters = 13,120
* Best Train Accuracy = 98.76
* Best Test Accuracy = 98.73
### Analysis
* Good model, no overfitting or underfitting
* We need to push our model to get better accuracy


## Second Model
[File Link](https://github.com/sridevibonthu/EVA/blob/master/S5/EVA4_S5_02.ipynb)

### Target
* Less than 20000 parameters
* Less than 15 epochs
* Using Batch_norm to improve accuracy
### Results
* Number of Parameters = 13,120 + 180 non trainable BN params
* Best Train Accuracy = 99.825
* Best Test Accuracy = 99.24
### Analysis
* Better accuracy but model is overfitting.
* We need to regularize
* We also have to reduce parameters to meet the assignment goal

## Third Model
[File Link](https://github.com/sridevibonthu/EVA/blob/master/S5/EVA4_S5_03.ipynb)

### Target
* Less than 10000 parameters
* Less than 15 epochs
* Add Image Augmentation (Rotation)
* Add GAP to reduce number of parameters
* Restructure the architecture to meet receptive field requirement
### Results
* Number of Parameters = 9034 + 140 non-trainable BN parameters
* Best Train Accuracy = 99.128
* Best Test Accuracy = 99.44
### Analysis
* As thought, model is no longer overfitting. Test accuracy is always above train accuracy
* There is lot of room to improve train accuracy still and some of it will get transferred to test accuracy.
* we can even try to reduce number of parameters and make the model more challenging.

## Fourth Model
[File Link](https://github.com/sridevibonthu/EVA/blob/master/S5/EVA4_S5_04.ipynb)

### Target
* Less than 10000 parameters
* Less than 15 epochs
* Add LR Scheduler with a step LR a gamma of 0.5 after every 5 steps
### Results
* Number of Parameters = 9034 + 140 non trainable BN params
* Best Train Accuracy = 99.21
* Best Test Accuracy = 99.55
### Analysis
* We bumped up the Training accuracy that reflected well in test accuracy
* Next we shall try to further reduce number of parameters, try dropouts (although the model suggests there is no need to but because this is one of the assignment requirments we are trying it). However Adding Dropout will make the model more challenging and give more room to improve the test accuracy.
* We shall try some more LR scheduler like cyclic LR. We will use minimum learnng rate of 0.1 (10 times more than what we used in stepLR)

## Fifth Model
[File Link](https://github.com/sridevibonthu/EVA/blob/master/S5/EVA4_S5_05.ipynb)

### Target
* Less than 8000 parameters
* Less than 15 epochs
* Test with Cyclic LR
* Add small dropout of 4%
### Results
* Number of Parameters = 7618 + 140 non trainable BN params
* Best Train Accuracy = 99.18
* Best Test Accuracy = 99.63
### Analysis
* As expected the cyclic LR worked better.
* Can increase dropouts more to further bump up the challenge. We are happy to be below 8000. But we can try to push the model further.

Bonus Model -- pushing the limits
File Link

Target
Less than 7000 parameters
Less than 15 epochs
Test with Cyclic LR
Add small dropout of 5%
Results
Number of Parameters = 6202 + 140 non trainable BN params
Best Train Accuracy = 98.97
Best Test Accuracy = 99.45
Analysis
We pushed the model to achieve target with approx 6000 Parameters
Drop in accuracy was predicted because of making the training more difficult. But still we met the target.
