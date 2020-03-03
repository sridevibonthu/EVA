# S7

We did all the required things:

1. We started with 400K odd parameters with 1 dilated conv in parallel and two depth separated conv. Got 80+ accuracy in 5th Epoch [link](https://github.com/abhinavdayal/EVA4/blob/master/S7/EVA04_S7_400K.ipynb)
2. Reduced to 120K params and got 80+ accuracy in 8 epochs. [link](https://github.com/abhinavdayal/EVA4/blob/master/S7/EVA04_S7_120K.ipynb)
3. Replaced all conv to depthwaise separable reducing to 40K params and got 80+ accuracy in 18 Epochs. [link](https://github.com/abhinavdayal/EVA4/blob/master/S7/EVA04_S7_40K.ipynb)
4. Moved dialated conv to main layer in first conv block. Removed depthwise except in last conv block. 100K params got 80+ accuracy in 9th Epoch. [link](https://github.com/abhinavdayal/EVA4/blob/master/S7/EVA04_S7_100K.ipynb)
5. Reduced dual conv in each block to single dialated conv with dilation of 2. 50K params. Reached 80+ accuracy in 20th Epoch. [link](https://github.com/abhinavdayal/EVA4/blob/master/S7/EVA04_S7_50K.ipynb)

**NOTE**: We separated code into files. The library files are in [this folder](https://github.com/abhinavdayal/EVA4/tree/master/S7/EVA4library).

* We used google drive to host these files and in colab we imported the drive.
* We also calculated dataset mean and stdev to do image normalization and applied several image transforms.
