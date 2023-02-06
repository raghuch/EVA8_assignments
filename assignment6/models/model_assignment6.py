import torch
import torch.nn as nn
import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self) -> None:
        super(myCNN, self).__init__()

        #input block
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) #output size = 32x32

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=0, stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
         ) #output size = 15x15

        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3,3), padding=0, dilation=2 ,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) #output size = 11x11

        #depthwise separable
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=0, bias=False, groups=256),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) #output size = 9x9

        self.convBlock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,3), padding=0, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        ) #output size = 4x4


        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) #output size = 1

        #output layer
        self.convBlock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False)
        )


    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.gap(x)
        x = self.convBlock6(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
