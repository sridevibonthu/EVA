# prediction of the depth and mask for the foreground object. 

**Description:** Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. This is an open problem, and here I will present, how it is solved. A custom Dataset is created in the last assignment. The approach followed to curate the custom dataset is presented [here](https://github.com/sridevibonthu/EVA/blob/master/S14/Readme.md).

## 1. Data Preparation

- The training data is curated on our own as part of previous assignment. [Readme](https://github.com/sridevibonthu/EVA/blob/master/S14/Readme.md), [Source Code](https://github.com/sridevibonthu/EVA/blob/master/S14/Final_Data_Creation.ipynb)
- Statistics on data were also calculated as part of previous assignemnt. [Source Code](https://github.com/sridevibonthu/EVA/blob/master/S14/Data_Statistics.ipynb)
- 100 background and 100 foreground images were used to generate 400K images of fgbg, mask and depth images. The selected foreground images are stray cows, calfs.
- The generated images are of size 224X224
- the 100 background images were placed in one folder and 400K fgbg, mask, depth images in their respective folders. 
- A datastructure is created which contains 400K records. each record has path to bg, fgbg, mask and depth like the following.

```
data = prepareData('./data')
print(len(data))
print(data[0])
print(data[39999])

400000
('./data/bgimages/bgimg072.jpg', './data/out2/images/fgbg285049.jpg', './data/out2/masks/mask285049.jpg', './data/out2/depth/fgbg285049.jpg')
('./data/bgimages/bgimg097.jpg', './data/out2/images/fgbg385867.jpg', './data/out2/masks/mask385867.jpg', './data/out2/depth/fgbg385867.jpg')
```
- the **prepareData** method takes in a folder which contains four folders and returns records which contains paths to the images. Every record has 4 paths, first two paths correspond to input (background and fgbg) and last two corresponds to targets(mask and depth)

- another utility **displayData** is written to print the images of the given record with the help of the paths. these utilities are available [here](https://github.com/sridevibonthu/EVALibrary/blob/master/EVA4/utils.py).
```
displayData(data, 14909)  #routine to display the respective record
```
![Sample Record](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/SAMPLE2.png)

## 2. Results

### Why am I presenting these so early?

The key point, I observed here is .... **"This assignment is an endless assignment."** I am keeping on getting ideas to change model structure, usage of loss function, schedulers, optimizer.

To predict mask and estimate depth, I have trained and tested several models, tried many loss functions and improved it step by step. Before I convey all those pains, I want to show one of my models outcome for which I got some satisfaction. [The Notebook, I have submitted](https://github.com/sridevibonthu/EVA/blob/master/S15/Final_Model_with_all_data_L1%2C_SSIM%2C_128X128_10_epochs_with_augmentions_good.ipynb)  

* The model is trained on 400K records with 70%-30% train-test split.
* With a batch size of 64, I have 4375 batches in my training data.
* The key features of the model are
1. My approach leverages encoder-decoder type architecture with skipped connections. but the change is it has two decoders
2. No.of parameters are reduced by employing deconvolution in downsampling. Parameters - 5 M
3. The parameter summary
```
model.summary(input_size=([(3,128,128),(3,128,128)]))

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 128, 128]           1,728
       BatchNorm2d-2         [-1, 32, 128, 128]              64
              ReLU-3         [-1, 32, 128, 128]               0
            Conv2d-4           [-1, 64, 64, 64]          18,432
       BatchNorm2d-5           [-1, 64, 64, 64]             128
              ReLU-6           [-1, 64, 64, 64]               0
            Conv2d-7           [-1, 64, 64, 64]           4,096
       BatchNorm2d-8           [-1, 64, 64, 64]             128
              ReLU-9           [-1, 64, 64, 64]               0
           Conv2d-10           [-1, 64, 64, 64]             576
      BatchNorm2d-11           [-1, 64, 64, 64]             128
             ReLU-12           [-1, 64, 64, 64]               0
           Conv2d-13           [-1, 64, 64, 64]           4,096
      BatchNorm2d-14           [-1, 64, 64, 64]             128
             ReLU-15           [-1, 64, 64, 64]               0
           Conv2d-16           [-1, 64, 64, 64]           2,048
      BatchNorm2d-17           [-1, 64, 64, 64]             128
             ReLU-18           [-1, 64, 64, 64]               0
     DownSampling-19           [-1, 64, 64, 64]               0
           Conv2d-20          [-1, 128, 32, 32]          73,728
      BatchNorm2d-21          [-1, 128, 32, 32]             256
             ReLU-22          [-1, 128, 32, 32]               0
           Conv2d-23          [-1, 128, 32, 32]          16,384
      BatchNorm2d-24          [-1, 128, 32, 32]             256
             ReLU-25          [-1, 128, 32, 32]               0
           Conv2d-26          [-1, 128, 32, 32]           1,152
      BatchNorm2d-27          [-1, 128, 32, 32]             256
             ReLU-28          [-1, 128, 32, 32]               0
           Conv2d-29          [-1, 128, 32, 32]          16,384
      BatchNorm2d-30          [-1, 128, 32, 32]             256
             ReLU-31          [-1, 128, 32, 32]               0
           Conv2d-32          [-1, 128, 32, 32]           8,192
      BatchNorm2d-33          [-1, 128, 32, 32]             256
             ReLU-34          [-1, 128, 32, 32]               0
     DownSampling-35          [-1, 128, 32, 32]               0
           Conv2d-36          [-1, 256, 16, 16]         294,912
      BatchNorm2d-37          [-1, 256, 16, 16]             512
             ReLU-38          [-1, 256, 16, 16]               0
           Conv2d-39          [-1, 256, 16, 16]          65,536
      BatchNorm2d-40          [-1, 256, 16, 16]             512
             ReLU-41          [-1, 256, 16, 16]               0
           Conv2d-42          [-1, 256, 16, 16]           2,304
      BatchNorm2d-43          [-1, 256, 16, 16]             512
             ReLU-44          [-1, 256, 16, 16]               0
           Conv2d-45          [-1, 256, 16, 16]          65,536
      BatchNorm2d-46          [-1, 256, 16, 16]             512
             ReLU-47          [-1, 256, 16, 16]               0
           Conv2d-48          [-1, 256, 16, 16]          32,768
      BatchNorm2d-49          [-1, 256, 16, 16]             512
             ReLU-50          [-1, 256, 16, 16]               0
     DownSampling-51          [-1, 256, 16, 16]               0
           Conv2d-52            [-1, 512, 8, 8]       1,179,648
      BatchNorm2d-53            [-1, 512, 8, 8]           1,024
             ReLU-54            [-1, 512, 8, 8]               0
           Conv2d-55            [-1, 512, 8, 8]         262,144
      BatchNorm2d-56            [-1, 512, 8, 8]           1,024
             ReLU-57            [-1, 512, 8, 8]               0
           Conv2d-58            [-1, 512, 8, 8]           4,608
      BatchNorm2d-59            [-1, 512, 8, 8]           1,024
             ReLU-60            [-1, 512, 8, 8]               0
           Conv2d-61            [-1, 512, 8, 8]         262,144
      BatchNorm2d-62            [-1, 512, 8, 8]           1,024
             ReLU-63            [-1, 512, 8, 8]               0
           Conv2d-64            [-1, 512, 8, 8]         131,072
      BatchNorm2d-65            [-1, 512, 8, 8]           1,024
             ReLU-66            [-1, 512, 8, 8]               0
     DownSampling-67            [-1, 512, 8, 8]               0
          Encoder-68  [[-1, 32, 128, 128], [-1, 64, 64, 64], [-1, 128, 32, 32], [-1, 256, 16, 16], [-1, 512, 8, 8]]               0
  ConvTranspose2d-69          [-1, 128, 32, 32]         294,912
           Conv2d-70          [-1, 128, 32, 32]         147,456
      BatchNorm2d-71          [-1, 128, 32, 32]             256
             ReLU-72          [-1, 128, 32, 32]               0
           Conv2d-73          [-1, 128, 32, 32]           1,152
      BatchNorm2d-74          [-1, 128, 32, 32]             256
             ReLU-75          [-1, 128, 32, 32]               0
           Conv2d-76          [-1, 128, 32, 32]          16,384
      BatchNorm2d-77          [-1, 128, 32, 32]             256
             ReLU-78          [-1, 128, 32, 32]               0
       UpSampling-79          [-1, 128, 32, 32]               0
  ConvTranspose2d-80           [-1, 64, 64, 64]          73,728
           Conv2d-81           [-1, 64, 64, 64]          36,864
      BatchNorm2d-82           [-1, 64, 64, 64]             128
             ReLU-83           [-1, 64, 64, 64]               0
           Conv2d-84           [-1, 64, 64, 64]             576
      BatchNorm2d-85           [-1, 64, 64, 64]             128
             ReLU-86           [-1, 64, 64, 64]               0
           Conv2d-87           [-1, 64, 64, 64]           4,096
      BatchNorm2d-88           [-1, 64, 64, 64]             128
             ReLU-89           [-1, 64, 64, 64]               0
       UpSampling-90           [-1, 64, 64, 64]               0
  ConvTranspose2d-91         [-1, 32, 128, 128]          18,432
           Conv2d-92         [-1, 32, 128, 128]           9,216
      BatchNorm2d-93         [-1, 32, 128, 128]              64
             ReLU-94         [-1, 32, 128, 128]               0
           Conv2d-95         [-1, 32, 128, 128]             288
      BatchNorm2d-96         [-1, 32, 128, 128]              64
             ReLU-97         [-1, 32, 128, 128]               0
           Conv2d-98         [-1, 32, 128, 128]           1,024
      BatchNorm2d-99         [-1, 32, 128, 128]              64
            ReLU-100         [-1, 32, 128, 128]               0
      UpSampling-101         [-1, 32, 128, 128]               0
          Conv2d-102         [-1, 32, 128, 128]           9,216
     BatchNorm2d-103         [-1, 32, 128, 128]              64
            ReLU-104         [-1, 32, 128, 128]               0
          Conv2d-105          [-1, 1, 128, 128]              32
     MaskDecoder-106          [-1, 1, 128, 128]               0
 ConvTranspose2d-107          [-1, 256, 16, 16]       1,179,648
          Conv2d-108          [-1, 256, 16, 16]         589,824
     BatchNorm2d-109          [-1, 256, 16, 16]             512
            ReLU-110          [-1, 256, 16, 16]               0
          Conv2d-111          [-1, 256, 16, 16]           2,304
     BatchNorm2d-112          [-1, 256, 16, 16]             512
            ReLU-113          [-1, 256, 16, 16]               0
          Conv2d-114          [-1, 256, 16, 16]          65,536
     BatchNorm2d-115          [-1, 256, 16, 16]             512
            ReLU-116          [-1, 256, 16, 16]               0
      UpSampling-117          [-1, 256, 16, 16]               0
 ConvTranspose2d-118          [-1, 128, 32, 32]         294,912
          Conv2d-119          [-1, 128, 32, 32]         147,456
     BatchNorm2d-120          [-1, 128, 32, 32]             256
            ReLU-121          [-1, 128, 32, 32]               0
          Conv2d-122          [-1, 128, 32, 32]           1,152
     BatchNorm2d-123          [-1, 128, 32, 32]             256
            ReLU-124          [-1, 128, 32, 32]               0
          Conv2d-125          [-1, 128, 32, 32]          16,384
     BatchNorm2d-126          [-1, 128, 32, 32]             256
            ReLU-127          [-1, 128, 32, 32]               0
      UpSampling-128          [-1, 128, 32, 32]               0
 ConvTranspose2d-129           [-1, 64, 64, 64]          73,728
          Conv2d-130           [-1, 64, 64, 64]          36,864
     BatchNorm2d-131           [-1, 64, 64, 64]             128
            ReLU-132           [-1, 64, 64, 64]               0
          Conv2d-133           [-1, 64, 64, 64]             576
     BatchNorm2d-134           [-1, 64, 64, 64]             128
            ReLU-135           [-1, 64, 64, 64]               0
          Conv2d-136           [-1, 64, 64, 64]           4,096
     BatchNorm2d-137           [-1, 64, 64, 64]             128
            ReLU-138           [-1, 64, 64, 64]               0
      UpSampling-139           [-1, 64, 64, 64]               0
 ConvTranspose2d-140         [-1, 32, 128, 128]          18,432
          Conv2d-141         [-1, 32, 128, 128]           9,216
     BatchNorm2d-142         [-1, 32, 128, 128]              64
            ReLU-143         [-1, 32, 128, 128]               0
          Conv2d-144         [-1, 32, 128, 128]             288
     BatchNorm2d-145         [-1, 32, 128, 128]              64
            ReLU-146         [-1, 32, 128, 128]               0
          Conv2d-147         [-1, 32, 128, 128]           1,024
     BatchNorm2d-148         [-1, 32, 128, 128]              64
            ReLU-149         [-1, 32, 128, 128]               0
      UpSampling-150         [-1, 32, 128, 128]               0
          Conv2d-151         [-1, 32, 128, 128]           9,216
     BatchNorm2d-152         [-1, 32, 128, 128]              64
            ReLU-153         [-1, 32, 128, 128]               0
          Conv2d-154          [-1, 1, 128, 128]              32
    DepthDecoder-155          [-1, 1, 128, 128]               0
================================================================
Total params: 5,525,568
Trainable params: 5,525,568
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 9216.00
Forward/backward pass size (MB): 256.00
Params size (MB): 21.08
Estimated Total Size (MB): 9493.08
----------------------------------------------------------------
```

* The model is trained for 10 epochs, i.e., 43750 batches. For every 2000 batches, I have saved the predictions in the [google drive](https://drive.google.com/drive/u/0/folders/1-3CZgEYGlepHIZCMhGZ80uRzrDcXH3p4) and also thrown to tensorboard. 

* **At 42000 step : Input** [FgonBg](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/fgbg42000.jpg), [Mask](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/orimask42000.jpg), [Depth](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/oridepth42000.jpg)

* **Depth estimation**
![Depth Estimation](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/preddepth42000.jpg)

* **Mask prediction**
![Mask Prediction](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/predmask42000.jpg)

* At the end of every epoch, a tensor is created with the eight images of fgbg, input mask, predicated mask, input depth, estimated depth in testing for the first batch.  

```
            if batch_idx == 0:
              inp = fgbg.detach().cpu()
              orimp = mask.detach().cpu()
              mp = mask_pred.detach().cpu()
              oridp = depth.detach().cpu()
              dp = depth_pred.detach().cpu()
              print("First batch in testing fgbg, (mask, predicted mask), (depth, predicted depth)")
              show(inp[:8,:,:,:], normalize=True)
              mdinp = torch.cat([orimp[:8,:,:,:], mp[:8,:,:,:], oridp[:8,:,:,:], dp[:8,:,:,:]],dim=0)
              show(mdinp)
```
![Epoch-10](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/Epoch%2010.png)

* Model pridictions on **Unseen** data. ![predications on unseen data](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/ModelPredictionsOnUnseenData.jpeg)

* The train loss is recorded for every 500 batches. ![train/loss](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/train%20loss.JPG)

## 3. Code Structure

- The library created by us from the past few assignments is used with few modifications. 
- Modular structure is followed to run the code. Training, Testing, model were designed as classes.
- My Library is [here](https://github.com/sridevibonthu/EVALibrary)

## 4. My efforts

* The dataset is prepared with paths to every image.
* Helper functions to display the data, show results grid, saving the images were written.
* A Custom Dataset Loader is designed quickly, and changes were made after running my models.
```
class CowDataset(Dataset):
  def __init__(self, data, bgtransforms, fgbgtransforms, masktransforms, depthtransforms):
    self.bg_files, self.fgbg_files, self.mask_files, self.depth_files = zip(*data)
    self.bgtransforms = bgtransforms
    self.fgbgtransforms = fgbgtransforms
    self.masktransforms = masktransforms
    self.depthtransforms = depthtransforms
    
  def __len__(self):
    return len(self.fgbg_files)
    
  def __getitem__(self, index):
    bg_image = Image.open(self.bg_files[index])
    fgbg_image = Image.open(self.fgbg_files[index])
    mask_image = Image.open(self.mask_files[index])
    depth_image = Image.open(self.depth_files[index])

    bg_image = self.bgtransforms(bg_image)
    fgbg_image = self.fgbgtransforms(fgbg_image)
    mask_image = self.masktransforms(mask_image)
    depth_image = self.depthtransforms(depth_image)
      
    return {'bg':bg_image, 'fgbg':fgbg_image, 'mask':mask_image, 'depth':depth_image}
```
* train and test datasets were created by objects of the above class
```
train = CowDataset(data[:trainlen],  bg_transforms, fgbg_transforms, mask_transforms, depth_transforms)
test = CowDataset(data[trainlen:], test_bgtransforms, test_fgbgtransforms, mask_transforms, depth_transforms)
print(len(train), len(test))

280000 120000
```
* There is no number left from 32 to 128, as a batch size for my trails, while loading data because of memory restrictions in my early runs.

## Model Creation and Experimentation with Loss functions

## A simple model

* Started with **a simple model** which has few convolutional layers with a Receptive Field of 22 and 3 Million parameters. This model has shown some fruitful outcome when it is trained only for masks with BCELossWithLogits, but same model when trained only for depth has given bad results.Results were not at all encouraging. I tried this same model with MSELoss, L1Loss. One more observation is MSELoss is vanishing at some point and I am getting complete blackend image as output.

**Mask Prediction** 

![Mask](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/bad%20result.png)

**Depth Pridiction**

![Depth](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/baddepthoutcome.png)

**Outcomes:**
1. improve the model.
2. Simpler model is taking more than one hour per epoch. Therefore i have to work on small amount of data at the beginning.
3. Reduce image size, so that I can send more input to model per batch.

## Resnet18 like model

* Tried **Resnet18** Architecture on masks and depths separately with 11 Million parameters. then made that model to output 2 X 1 X W X H as two outcomes. This is a bad idea and the results are almost similar and not good.

![outcomes](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/Resnet1.jpeg)

**Outcome:**
1. Plan different model architecture 

## Encoder - Decoder architecture

* Followed Unet paper and implemented an **Encoder-Decoder** kind of architecure. Here the challenge to me is how to write decoder. I used Maxunpooling, 1X1 convolutions, transposed convolutions alone, transposed convolutions and interpolation with bilinear mode. Among all Maxunpooling has given worse results. I had two decoders here. Trained this network with few loss functions. L1, MSE, BCE combinations were tested by me. Next i used SSIM. No loss function alone worked for me.

Use of Maxunpooling has given 
![maxunpooling](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/Maxunpooling.jpeg)

Some more bad results which made me try different loss functions are
|![one](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/encdec1.png) |![two](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/encdec2.png) |![three](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/encdec3.png) | ![Four](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/encoder.jpeg) |

**My Model:**

```
class S15Model(Net):
  def __init__(self):
    super(S15Model, self).__init__()
    self.encoder = Encoder()
    self.decoder1 = Decoder()
    self.decoder2 = Decoder()
    self.init_params()
        
  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    e = self.encoder(x)
    mask = self.decoder1(*e)
    depth = self.decoder2(*e)
    return(mask, depth)
```
**Encoder** class
```
class Encoder(Net):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = self.create_conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.down1= DownSampling(32, 64)
    self.down2 = DownSampling(64,128)
    self.down3 = DownSampling(128, 256)
    self.down4 = DownSampling(256, 512)
    
  def forward(self,x):
    out0 = self.conv1(x)
    out1 = self.down1(out0)
    out2 = self.down2(out1)
    out3 = self.down3(out2)
    out4 = self.down4(out3)
    return out0, out1, out2, out3, out4
 ```
 **Decoder** Class
 ```
 class Decoder(Net):
  def __init__(self):
    super(Decoder, self).__init__()
    self.up1 = UpSampling(512,256)
    self.up2 = UpSampling(256, 128)
    self.up3 = UpSampling(128,64)
    self.up4 = UpSampling(64, 32)
    self.convend1 = self.create_conv2d(32, 32, kernel_size=3, padding=1)
    self.convend2 = self.create_conv2d(32, 1, kernel_size=1, padding=0, bn=False, relu=False)
    
        
  def forward(self, x0, x1, x2, x3,x4):
    y = x3 + self.up1(x4)
    y = x2 + self.up2(x3)
    y = x1 + self.up3(y)
    y = x0 + self.up4(y)
    y = self.convend1(y)
    y = self.convend2(y)
    return(y)
```

**outcome**
1. Reduce the number of parameters.
2. work on loss function

## Improved Model with SSIM Loss

* Used dialated convolutions in one of the convolution layer of downsampling block to reduce number of parameters. My model landed at 8 Million parameters. with the help of the paper titled [Loss Functions for image Restoration with Neural Networks](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf), i combined ssim and another loss function (L1, MSE, BCE) and tested. I got best results when SSIM is combined with L1Loss. **(0.84 * SSIM + 0.16 * L1)**. I have also given more weight to Mask loss in the overall loss calculation.

```
            # Calculate loss
            if self.criterion1 is not None:
              loss1 = self.criterion1(mask_pred, mask)
            m_ssim = torch.clamp((1 - ssim(mask_pred, mask, normalize=True)) * 0.5, 0, 1)
            loss1 = (0.84 * m_ssim) + (0.16 * loss1)

            if self.criterion2 is not None:
                loss2 = self.criterion2(depth_pred, depth)
            d_ssim = torch.clamp((1 - ssim(depth_pred, depth, normalize=True)) * 0.5, 0, 1)
            loss2 = (0.84 * d_ssim) + (0.16 * loss2)

            self.loss = 2 * loss1 + loss2
```
**Outcome:** 
1. Things are going on smoothly and planned to submit.   (Hurray! Deadline is extended)

* **Next Model** : As I understood that Mask can be predicted with less receptive field, I reduced depth of mask decoder and now number of parameters of my model are 5 M. Results are pretty awesome here. I used SGD optimizer, One Cycle Policy to run this. I have submitted this model.

* **Next** - I have some more time. I experimented with MSSSIM Loss. It took time for me, as it used to vanish after few batches. After fixing this problem, MSSSIM Loss works more fine masks than depths. Number of parameters are further reduced to 3 Million by introducing dialated convoltions in both upsampling and downsampling. 

* Few othe measures I have take while doing this work
1. The best model is saved whenever a validation loss better than best loss so far is obtained to come out of the restrictions of Google colab limited time with the help of the following code block
```
    if self.epochs == 1 or self.best_test_loss > self.avg_test_loss[-1]:
      print(f'Validation loss decreased ({self.best_test_loss:.6f} --> {self.avg_test_loss[-1]:.6f}).  Saving model ...')
      torch.save(self.model.state_dict(), f"{self.path}/{self.model.name}.pt")
      self.best_test_loss = self.avg_test_loss[-1]
```
![SavingModel](https://github.com/sridevibonthu/EVA/blob/master/S15/Images/saving%20model.JPG)

2. Measured time taken for data loading, per epoch how much time loss, model, plotting is taking. By reducing data augmentations, i reduced time taken for data loading. As there is not much difference between test and train losses, i decided to reduce augmentaions. Done intermediate plotting once per 3000 batches.
```
      n = self.stats.get_batches()
      if (n+1)%500 == 0:
        self.tb.add_scalar('loss/train', loss.item(), n)
        self.tb.add_scalar('Leaning Rate', self.stats.batch_lr[-1])
      
      if (n+1) % 3000 == 0:
        grid = torchvision.utils.make_grid(mask_pred.detach().cpu(), nrow=8, normalize=False)
        self.tb.add_image('imagesmask', grid, n)
        grid = torchvision.utils.make_grid(depth_pred.detach().cpu(), nrow=8, normalize=False)
        self.tb.add_image('imagesdepth', grid, n)
      
      
        saveresults(fgbg.detach().cpu(), "./plots/fgbg"+str(n+1)+".jpg", normalize=True)
        saveresults(mask.detach().cpu(), "./plots/orimask"+str(n+1)+".jpg")
        saveresults(depth.detach().cpu(), "./plots/oridepth"+str(n+1)+".jpg")
        saveresults(mask_pred.detach().cpu(), "./plots/predmask"+str(n+1)+".jpg")
        saveresults(depth_pred.detach().cpu(), "./plots/preddepth"+str(n+1)+".jpg")
```
3. Planned to calculate accuracy, by finding difference between the input and predictions. But i have not implemented it.

My Journey started from **a simple model** with 3 Million parameters with BCE, MSE loss and ended at an **encoder-decoder model** with 3 million parameters which employs L1 Loss + SSIM.


Thanks to **Rohan Sravan** for his great mentorship and all my team who helped from week 1 to 14.





