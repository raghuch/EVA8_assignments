import torch
import torch.nn as nn
import torch.nn.functional as F


class UltimusBlock(nn.Module):
    def __init__(self, in_feat, out_feat) -> None:
        super(UltimusBlock, self).__init__()
        self.k_dims = out_feat #dimensions of k, here dims_k = 8
        self.k_fc = nn.Linear(in_features=in_feat, out_features=out_feat)
        self.q_fc = nn.Linear(in_features=in_feat, out_features=out_feat)
        self.v_fc = nn.Linear(in_features=in_feat, out_features=out_feat)

        self.fc = nn.Linear(8, 48)


    def _get_am(self, q, k):
        return F.softmax(((q.transpose(1, 2)@k)/(self.k_dims**0.5)) ,dim=1 )
    
    def _get_z(self, am, v):
        return v@am #matmul of v and am


    def forward(self, x):
        k = self.k_fc(x)
        k = k.view(k.size(0), 1, -1)
        q = self.q_fc(x)
        q = q.view(q.size(0), 1, -1)
        v = self.v_fc(x)
        v = v.view(v.size(0), 1, -1)
        am = self._get_am(q, k)
        z = self._get_z(am, v)

        return self.fc(z)
    
class conv_relu_bn_dropout_blk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sz, dropout, stride=1, padding=1, dilation=1) -> None:
        super(conv_relu_bn_dropout_blk, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sz, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.conv(x)
    

class BasicTransformer(nn.Module):
    dropout = 0.1
    def __init__(self):
        super(BasicTransformer, self).__init__()
        self.conv_blk = nn.Sequential(
            conv_relu_bn_dropout_blk(3, 16, (3,3), self.dropout, padding=1),
            conv_relu_bn_dropout_blk(16, 32, (3,3), self.dropout, padding=1),
            conv_relu_bn_dropout_blk(32, 48, (3,3), self.dropout, padding=1)
        ) #in size = out size

        self.gap1 = nn.AdaptiveAvgPool2d(1) #Use adaptive vs avg pooling, argument is the output dimension
        # output is 1x1x48
        

        self.ultimus = nn.Sequential(
            UltimusBlock(in_feat=48, out_feat=8),
            UltimusBlock(in_feat=48, out_feat=8),
            UltimusBlock(in_feat=48, out_feat=8),
            UltimusBlock(in_feat=48, out_feat=8)
        )

        self.ffc = nn.Linear(48, 10)


    # def _make_conv_layer(conv_blk, in_channels, out_channels, kernel_sz, padding, stride=1):
    #     layers = []


    def forward(self, x):
        x = self.conv_blk(x)
        x = self.gap1(x)
        x = x.view(-1, 48)
        x = self.ultimus(x)
        out = self.ffc(x)

        out = out.view(out.size(0), -1)
        return out


def get_ultimus():
    return BasicTransformer()