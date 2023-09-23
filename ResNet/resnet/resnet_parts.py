import torch 
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, num_layers,in_channels,out_channels, stride = 1, expansion= 1,downsample:nn.Module = None):
        super(Block, self).__init__()
        self.num_layers = num_layers
        self.expansion = expansion
        self.downsample = downsample

        if self.num_layers in [18,34]:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            
            self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels*self.expansion, kernel_size = 3, stride = 1, padding = 1, bias = False )
            self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
            self.relu = nn.ReLU(inplace=True)
           
        
        else: 
            self.conv0 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1,stride = 1,  bias = False)
            self.bn0 = nn.BatchNorm2d(out_channels)
            in_channels = out_channels

            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,stride = stride, padding = 1, bias = False)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels * self.expansion, kernel_size = 1, stride = 1, bias = False)
            self.bn2 = nn.BatchNorm2d(out_channels* self.expansion )
            self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        identity = x.clone()
        if self.num_layers in [18,34]:
            output = self.conv1(x)
            output = self.bn1(output)
            

            output = self.conv2(output)
            output = self.bn2(output)
            output = self.relu(output)
        else:
            output = self.conv0(x)
            output = self.bn0(output)
            output = self.relu(output)

            output = self.conv1(output)
            output = self.bn1(output)
            output = self.relu(output)

            output = self.conv2(output)
            output = self.bn2(output)
            
        if self.downsample is not None:
            identity = self.downsample(x)
        output += identity
        output = self.relu(output)
        
        return output
