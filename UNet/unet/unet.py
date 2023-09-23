import torch 
from torch import nn
from torch.nn import functional as F
from unet.unet_parts import Block, Downsample, Upsample


    

class UNet(nn.Module):
    def __init__(self, img_channels,n_classes):
        super(UNet,self).__init__()
        
        self.img_channels = img_channels
        self.out_channels = 64
        self.expansion = 2
        self.n_classes = n_classes

        # Left-side of U
        self.layer1 = Block(in_channels = self.img_channels, out_channels = self.out_channels) # 3 - 64
        self.layer2 = Downsample(in_channels = self.out_channels, out_channels = self.out_channels * self.expansion) # 64 - 128
        
        self.out_channels = self.out_channels * self.expansion
        self.layer3 = Downsample(in_channels = self.out_channels, out_channels = self.out_channels * self.expansion) # 128 - 256

        self.out_channels = self.out_channels * self.expansion
        self.layer4 = Downsample(in_channels = self.out_channels, out_channels = self.out_channels * self.expansion) # 256 - 512

        self.out_channels = self.out_channels * self.expansion
        self.layer5 = Downsample(in_channels = self.out_channels, out_channels = self.out_channels * self.expansion // self.expansion) # 512 - 512

        # Right-side of U
        self.layer6 = Upsample(in_channels = self.out_channels * self.expansion, out_channels = self.out_channels // self.expansion) # 1024 - 256

        
        self.layer7 = Upsample(in_channels = self.out_channels , out_channels = self.out_channels // (self.expansion * self.expansion)) # 512 - 128

        self.out_channels = self.out_channels // self.expansion
        self.layer8 = Upsample(in_channels = self.out_channels , out_channels = self.out_channels // (self.expansion * self.expansion)) # 256 - 64

        self.out_channels = self.out_channels // self.expansion
        self.layer9 = Upsample(in_channels = self.out_channels , out_channels = self.out_channels // self.expansion) # 128 - 64
    
        # Output channel
        self.out_channels = self.out_channels // self.expansion
        self.outc = nn.Conv2d(self.out_channels, self.n_classes, kernel_size=1)


       # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
       # self.fc = nn.Linear(512*self.expansion, num_classes)
    
                
    def forward(self, x):
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x5 = self.layer5(x4)
            x = self.layer6(x5,x4)
            x = self.layer7(x,x3)
            x = self.layer8(x,x2)
            x = self.layer9(x,x1)
            x = self.outc(x)
            
            return x
