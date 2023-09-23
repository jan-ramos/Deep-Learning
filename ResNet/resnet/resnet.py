import torch 
from torch import nn
from torch.nn import functional as F
from resnet_parts import Block

layers = {18:[2,2,2,2],34:[3,4,6,3],50:[3,4,6,3],101:[3, 4, 23, 3],152:[3, 8, 36, 3]}

class ResNet(nn.Module):
    def __init__(self, img_channels,num_layers,num_classes:1000):
        super(ResNet,self).__init__()
        
        self.layers = layers[num_layers]
        if num_layers  in [18,34]:
            self.expansion = 1
        else:   
            self.expansion = 4

        self.in_channels = 64

        # All layers begin with a Convolution layer with a 7x7 kernel and stride 2

        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # We then apply a Max Pooling layer with 3x3 kernel and stride 2 

        self.max = nn.MaxPool2d(kernel_size = 3,stride = 2 ,padding = 1)
        
        # We then build the ResNet layers based on the number of layers 
    
        self.layer1 = self.create_layer(out_channels = 64, blocks = self.layers[0], stride = 1, num_layers=num_layers)
        

        self.layer2 = self.create_layer(out_channels = 128, blocks = self.layers[1], stride=2, num_layers=num_layers)
        self.layer3 = self.create_layer(out_channels = 256, blocks = self.layers[2], stride=2, num_layers=num_layers)
        self.layer4 = self.create_layer(out_channels = 512, blocks = self.layers[3], stride=2, num_layers=num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    
    def create_layer(self,out_channels,blocks,stride= 1,num_layers= 18) -> nn.Sequential:
        downsample = None
        if stride != 1  or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion,kernel_size=1, stride=stride,bias=False ),
                nn.BatchNorm2d(out_channels*self.expansion),
            )
        layers = []
        layers.append(Block(num_layers, self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion
        
        for i in range(1, blocks):
            layers.append(Block(num_layers,self.in_channels,out_channels,expansion=self.expansion))
        
        return nn.Sequential(*layers)
                
    def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.max(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # The spatial dimension of the final layer's feature 
            # map should be (7, 7) for all ResNets.
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x