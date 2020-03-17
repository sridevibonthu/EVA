# S9
We worked as a team. 

## Team Members
1. B Sridevi  - sridevi.b@vishnu.edu.in
2. Abhinav Dayal - abhinav.dayal@vishnu.edu.in
3. A. Lakshmana Rao - 18pa1a0511@vishnu.edu.in
4. Sanjay Varma G - 18pa1a1211@vishnu.edu.in

[My Final Notebook]()
* We have gone through albumentations library and experimented with cutout, horizontal flip, rgbshift, and rotate along with usual Normalize and ToTensor.
* Added these in train transforms and test transforms is only Normalize and ToTensor
* We implemented GradCam with the help of the library .. https://github.com/kazuto1011/grad-cam-pytorch
* We tested the GradCam in various ways. We have not only tested at the end of fourth layer (before gap), but at the end of every layer. Visualization of all 4 layer outputs of Resnet are present in our notebook.

My submission contains 3 modules 
1. [Data Transforms](https://github.com/sridevibonthu/EVA/blob/master/S9/eva4datatransforms.py) - added albumentation code
2. [Gradcam](https://github.com/sridevibonthu/EVA/blob/master/S9/eva4gradcam.py) module
3. [QuizDNN](https://github.com/sridevibonthu/EVA/blob/master/S9/QuizDNN.py) module
