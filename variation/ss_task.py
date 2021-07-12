import torch.nn as nn
import math
import torch
from torch.nn.functional import sigmoid, softmax, relu

class ss_task(nn.Module):
    def __init__(self):
        super(ss_task, self).__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=100, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(100*7*7,  7*7)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.conv.weight)
        
        #suggested by professor
        #feat (BS,512x7x7) --> relu() = feat (BS,512x7x7)
        #feat (BS,512x7x7) --> CONV[100, k=1, p=0] = feat(BS,100x7x7)
        #feat(BS,100x7x7) --> feat(BS,100*7*7)
        #feat(BS,100*7*7) --> FC[100*7*7, 2*7*7] = feat(BS, 2*7*7) #the 2 is for classification
        #feat(BS, 2*7*7) --> feat(BS, 2x7x7)
        #cross_entropy(feat, maps)
        
    def forward(self, x):
        out = torch.relu(x)
        out = self.conv(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out
