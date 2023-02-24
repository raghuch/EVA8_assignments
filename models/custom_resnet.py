# Custom ResNet for EVA8 assignment-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, padding=1) -> None:
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=(3,3),stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3,3), stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1 ,stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out) 

        return out
    
class customResNet(nn.Module):
    def __init__(self, resblock, n_blocks, n_classes=10) -> None:
        super(customResNet, self).__init__()
        self.in_channels = 64
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) #out size  = in size
        self.X1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) #out size = halved

        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # out size = halved
        self.X2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) #out size = halved
        
        self.R1 = self._make_resnet_layer(resblock, 64, 128, n_blocks[0], stride=2)
        self.R2 = self._make_resnet_layer(resblock, 256, 512, n_blocks[1], stride=2)

        self.maxpool4 = nn.MaxPool2d(4)
        self.fc = nn.Linear(512*resblock.expansion, out_features=n_classes, bias=False)

        self.flatten = nn.Flatten()



    # def _make_resnet_layer(self, resblock, channels, n_blocks, stride=1):
    #     expansion=1
    #     strides = [stride]+ [1]*(n_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(resblock(self.in_channels, channels, stride))
    #         self.in_channels = channels * resblock.expansion

    #     return nn.Sequential(*layers)


    def _make_resnet_layer(self, resblock, in_channels, out_channels, n_blocks, stride=1):
        expansion=1
        strides = [stride]+ [1]*(n_blocks-1)
        layers = []
        for stride in strides:
            layers.append(resblock(in_channels, out_channels, stride))
            #self.in_channels = channels * resblock.expansion

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.prep(x)
        l1x = self.X1(x)
        l1r1 = self.R1(x)
        out = l1x + l1r1
        out = self.l2(out)
        out = self.maxpool4(self.X2(out) + self.R2(out))
        out = self.flatten(out)
        out = self.fc(out)

        return F.log_softmax(out)
    
def get_customResNet():
    return customResNet(ResBlock, [1, 1])

    
