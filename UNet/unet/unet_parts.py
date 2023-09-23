import torch 
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,in_channels,out_channels, stride = 1, expansion= 1,downsample:nn.Module = None):
        super(Block, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv0 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding  = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        output = self.conv0(x)
        output = self.bn0(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu(output)

        return output
    
class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Downsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maxpool0 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv0 = Block(in_channels = self.in_channels, out_channels = self.out_channels)
        
    def forward(self,x):
        output = self.maxpool0(x)
        output = self.conv0(output)
        
        return output

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
       
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
       
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x