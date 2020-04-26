# Assignment 13 - YoloV3
________

## Part - A
OpenCV Yolo: ![link](https://github.com/sridevibonthu/EVA/blob/master/S13/EVA4_S13_Task1_OPENCV_YOLO.ipynb)
**Task** : 
* Run the code.
* Take an image of myself, holding another object which is there in COCO data set.
* Run this image through the code above. 
**Annotated Image**
![image](https://github.com/sridevibonthu/EVA/blob/master/S13/annotated_sridevi.png)


## Part - B
YoloV3 Simplified for training on Colab with custom dataset for one class (**GUN**)

_A Collage of Training images_
![image](https://github.com/sridevibonthu/YoloV3/blob/master/train_batch0.png)

_A Collage of Testing images_
![image](https://github.com/sridevibonthu/YoloV3/blob/master/test_batch0.png)

Class - gun

1. We have added a 500 images of unique object (gun) in the folder customdata after annotating the images using Annotation Tool. The structure we followed to store them is
```
data
  --customdata
    --images/
      --img001.jpg
      --img002.jpg
      --...
    --labels/
      --img001.txt
      --img002.txt
      --...
    custom.data #data file
    custom.names #class name
    customtrain.txt #list of name of the images to train our network.
    customtest.txt #list of names of the images for validation
```
2. For one class example our custom.data is [here](https://github.com/sridevibonthu/YoloV3/blob/master/data/customdata/custom.data). We used 500 images for training and 100 images for testng.
2. downloaded the weights (yolov3-spp-ultralytics.pt) from the original ![source](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0) and placed in Google Drive. 
3. Created a weights folder under YoloV3 to store weights
4. Trained for 300 epochs after configuring. (log)[https://github.com/sridevibonthu/YoloV3/blob/master/results.txt)



**Results**

After training for 300 Epochs, results look awesome!

![image](https://github.com/sridevibonthu/YoloV3/blob/master/output/img080.jpg)

![image](https://github.com/sridevibonthu/YoloV3/blob/master/output/img082.jpg)

**Performance**

![image](https://github.com/sridevibonthu/YoloV3/blob/master/results.png)
