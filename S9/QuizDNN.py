import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4models import Net

class QuizNet(Net):
    def __init__(self, name="Model", dropout_value=0):
        super(QuizNet, self).__init__(name)
		
		
        self.conv0 = self.create_conv2d(3, 32, dropout=dropout_value)  
        self.conv1= self.create_conv2d(32, 32, dropout=dropout_value) 
        self.conv2 = self.create_conv2d(32, 32, dropout=dropout_value) 

        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3_0 = self.create_conv2d(32, 64, kernel_size=(1,1),padding=0)
        self.conv3 = self.create_conv2d(32, 64, dropout=dropout_value) 
        self.conv4 = self.create_conv2d(64, 64, dropout=dropout_value) 
        self.conv5 = self.create_conv2d(64, 64, dropout=dropout_value)

        self.pool2 = nn.MaxPool2d(2, 2) 
        self.conv6_0 = self.create_conv2d(64, 128, kernel_size=(1,1),padding=0)
        self.conv6 = self.create_conv2d(64, 128, dropout=dropout_value)
        self.conv7 = self.create_conv2d(128, 128, dropout=dropout_value)
        self.conv8 = self.create_conv2d(128, 128, dropout=dropout_value) 

        self.pool3 = nn.MaxPool2d(2, 2)         
        
        # GAP + FC
        #self.gap = nn.AvgPool2d(kernel_size=(4,4)) 
        self.conv9 = self.create_conv2d(128, 10, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN: 256 OUT:10

    def forward(self, x):
      x1 = self.conv0(x)          #x1 = Input
      x2 = self.conv1(x1)      # x2 = Conv(x1)
      x2 = x1 + x2
      x3 = self.conv2(x2)      # x3 = Conv(x1 + x2)

      x3 = x2 + x3
      x4 = self.pool1(x3)       # x4 = MaxPooling(x1 + x2 + x3)

      x4_0 = self.conv3_0(x4) 
      x5 = self.conv3(x4)   # x5 = Conv(x4)  and transition from 32 to 64
      x5 = x4_0 + x5
      x6 = self.conv4(x5)       #x6 = Conv(x4 + x5)
      x6 = x5 + x6
      x7 = self.conv5(x6)       # x7 = Conv(x4 + x5 + x6)

      x8 = self.pool2(x7)       # x8 = MaxPooling(x5 + x6 + x7)
      x8_0 = self.conv6_0(x8)
      
      x9 = self.conv6(x8)       # x9 = Conv(x8)
      x9 = x8_0 + x9
      x10 = self.conv7(x9)      # x10 = Conv (x8 + x9)
      x10 = x9 + x10
      x11 = self.conv8(x10)     # x11 = Conv (x8 + x9 + x10)
        
      x12 = F.adaptive_avg_pool2d(x11, 1)   # x12 = GAP(x11)
      y = self.conv9(x12)        # x13 = FC(x12)   here X13 is y

      y = y.view(-1, 10)
      return F.log_softmax(y, dim=-1)
