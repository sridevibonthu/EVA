from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4modeltrainer import ModelTrainer

class Net(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]

    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True):
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))
      if relu:
        l.append(nn.ReLU())

      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)

    def __init__(self, name="Model"):
        super(Net, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None

class Cfar10Net(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 32, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x32, RF = 3
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 32x32x32, OUT 32x32x32, RF = 5
        self.conv3 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 32x32x32, OUT 32x32x32, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 16x16x32, OUT 16x16x64, RF = 12
        self.conv5 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 16x16x64, OUT 16x16x64, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv6 = self.create_conv2d(64, 128, dropout=dropout_value) # IN 8x8x64, OUT 8x8x128, RF = 26
        self.conv7 = self.create_conv2d(128, 128, dropout=dropout_value) # IN 8x8x128, OUT 8x8x128, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        
        self.conv8 = self.create_depthwise_conv2d(128, 256, dropout=dropout_value) # IN 4x4x128, OUT 4x4x256, RF = 54
        self.conv9 = self.create_depthwise_conv2d(256, 256, dropout=dropout_value) # IN 4x4x256, OUT 4x4x256, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(256, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net2(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(Cfar10Net2, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 5
        self.conv3 = self.create_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_conv2d(32, 64, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net3(Net):
    def __init__(self, name="Cfar10Net3", dropout_value=0):
        super(Cfar10Net3, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_depthwise_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_depthwise_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 5
        self.conv3 = self.create_depthwise_conv2d(16, 16, dropout=dropout_value) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_depthwise_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_depthwise_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.dconv1 = self.create_depthwise_conv2d(32, 64, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64
        self.conv6 = self.create_depthwise_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_depthwise_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x2 = self.dconv1(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.add(x, x2)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



class Cfar10Net4(Net):
    def __init__(self, name="Cfar10Net4", dropout_value=0):
        super(Cfar10Net4, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value, dilation=2, padding=2) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value) # IN 16x16x16, OUT 16x16x32, RF = 12
        self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value) # IN 8x8x32, OUT 8x8x64, RF = 26
        self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 54
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 70

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Cfar10Net5(Net):
    def __init__(self, name="Cfar10Net5", dropout_value=0):
        super(Cfar10Net5, self).__init__(name)

        # Input Convolution: C0
        self.conv1 = self.create_conv2d(3, 16, dropout=dropout_value)  # IN 32x32x3, OUT 32x32x16, RF = 3
        self.conv2 = self.create_conv2d(16, 16, dropout=dropout_value, dilation=2, padding=2) # IN 32x32x16, OUT 32x32x16, RF = 7

        # Transition 1
        self.pool1 = nn.MaxPool2d(2, 2) # IN 32x32x32 OUT 16x16x32, RF = 8, jump = 2

        self.conv4 = self.create_conv2d(16, 32, dropout=dropout_value, dilation=2, padding=2) # IN 16x16x16, OUT 16x16x32, RF = 16
        #self.conv5 = self.create_conv2d(32, 32, dropout=dropout_value) # IN 16x16x32, OUT 16x16x32, RF = 16

        # Transition 2
        self.pool2 = nn.MaxPool2d(2, 2) # IN 16x16x64 OUT 8x8x64, RF = 18, jump = 4
        self.conv6 = self.create_conv2d(32, 64, dropout=dropout_value, dilation=2, padding=2) # IN 8x8x32, OUT 8x8x64, RF = 34
        #self.conv7 = self.create_conv2d(64, 64, dropout=dropout_value) # IN 8x8x64, OUT 8x8x64, RF = 34

        # Transition 3
        self.pool3 = nn.MaxPool2d(2, 2) # IN 8x8x128 OUT 4x4x128, RF = 38, jump = 8
        #self.dconv2 = self.create_conv2d(64, 128, dilation=2, padding=2) # IN 8x8x64, OUT 8x8x128
        self.conv8 = self.create_depthwise_conv2d(64, 128, dropout=dropout_value) # IN 4x4x64, OUT 4x4x128, RF = 70
        self.conv9 = self.create_depthwise_conv2d(128, 128, dropout=dropout_value) # IN 4x4x128, OUT 4x4x128, RF = 86

        # GAP + FC
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv10 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.pool1(x)
        x = self.conv4(x)
        #x = self.conv5(x)

        x = self.pool2(x)
        x = self.conv6(x)
        #x = self.conv7(x)

        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = self.gap(x)
        x = self.conv10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class MnistNet(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(MnistNet, self).__init__(name)

        # Input Convolution Block
        self.convblock1 = self.create_conv2d(3, 32, dropout=dropout_value, groups=3) # input_size = 28, output_size = 28, RF = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = self.create_conv2d(10, 10, dropout=dropout_value) # output_size = 28, RF = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6

        self.convblock3 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 12, RF = 10

        # CONVOLUTION BLOCK 2
        self.convblock4 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 10, RF = 14

        self.convblock5 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 8, RF = 18

        self.convblock6 = self.create_conv2d(10, 10, dropout=dropout_value, padding=0) # output_size = 6, RF = 22

        self.convblock7 = self.create_conv2d(10, 16, dropout=dropout_value, padding=0, bn=False, relu=False) # output_size = 4, RF = 26

        # OUTPUT BLOCK
        self.gap = nn.AvgPool2d(kernel_size=(4,4)) 

        self.convblock8 = self.create_conv2d(16, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # output_size = 1, RF = 26
        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)