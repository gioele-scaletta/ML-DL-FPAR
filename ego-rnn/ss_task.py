import torch.nn as nn
import math
import torch
from torch.nn.functional import sigmoid

def conv1x1(in_planes, out_planes, kernel_size=1, stride=1, padding=0):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class ss_task(nn.Module):
    def __init__(self):
        super(ss_task, self).__init__()
        self.conv = conv1x1(in_planes=512, out_planes=100, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(100*7*7,  2*7*7)
        
        #suggested by professor
        #feat (BS,512x7x7) --> relu() = feat (BS,512x7x7)
        #feat (BS,512x7x7) --> CONV[100, k=1, p=0] = feat(BS,100x7x7)
        #feat(BS,100x7x7) --> feat(BS,100*7*7)
        #feat(BS,100*7*7) --> FC[100*7*7, 2*7*7] = feat(BS, 2*7*7)
        #feat(BS, 2*7*7) --> feat(BS, 2x7x7)
        #cross_entropy(feat, maps)
        
    def forward(self, x):
        out = torch.sigmoid(x)
        out = self.conv(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out
   
