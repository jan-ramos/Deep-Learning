import torch 
from torch import nn
from torch.nn import functional as F
from resnet import ResNet

if __name__ == '__main__':
    tensor = torch.rand([1, 3, 224, 224])
    model = ResNet(img_channels=3, num_layers=152, num_classes=1000)
    print(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output = model(tensor)