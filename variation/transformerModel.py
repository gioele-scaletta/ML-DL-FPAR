import torch
import resnetMod
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyTransformer import *


class selfAttentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(selfAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.transf = MyTransformer()
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.conv_f = nn.Conv1d(512, 61, kernel_size=1, stride=1)
        self.fc = nn.Linear(self.mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        nn.init.xavier_normal_(self.conv_f.weight)
        nn.init.constant_(self.conv_f.bias, 0)

    def forward(self, inputVariable):
        logits = []
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            n_frames, n_channels, h, w = feature_conv.size()	# n_channels = 512 and h x w = 7x7
            embedding = torch.squeeze(torch.squeeze(self.avgpool(feature_conv),3),2)
            logit = self.transf(embedding)
            final_logit = self.fc(logit)
            logits.append(final_logit.view(1,-1))
        logits = torch.stack(logits,axis=0).squeeze(1)
        return logits
    
