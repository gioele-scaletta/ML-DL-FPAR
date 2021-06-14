import torch
import AdaptedMobileNetV2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyTransformer2 import *


class selfAttentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=1280):
        super(selfAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.mobileNet = AdaptedMobileNetV2.mobilenet_v2(True, True)
        self.mem_size = 1280
        self.weight_softmax = list(self.mobileNet.classifier.children())[1].weight
        self.transf = MyTransformer()
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.conv_f = nn.Conv1d(1280, 61, kernel_size=1, stride=1)
        self.fc = nn.Linear(self.mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        nn.init.xavier_normal_(self.conv_f.weight)
        nn.init.constant_(self.conv_f.bias, 0)

    def forward(self, inputVariable):
        logits = []
        for t in range(inputVariable.size(0)):
            # prepare CAM as for resnet but now use last layer of mobileNetV2
            logit, feature_conv = self.mobileNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()    #bz = batch size    #nc = number of channels    #h = height   #w = width
            feature_conv1 = feature_conv.view(bz, nc, h*w)    #make vector from matrix (hxw -> h*w)
            probs, idxs = logit.sort(1, True)   #sort in order from highest to lowest the score of each class in each batch
            class_idx = idxs[:, 0]    
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_conv * attentionMAP.expand_as(feature_conv)
            
            # as input for the transformer use the features after attention
            embedding = torch.squeeze(torch.squeeze(self.avgpool(attentionFeat),3),2)
            logit = self.transf(embedding)
            final_logit = self.fc(logit.cuda())
            logits.append(final_logit.view(1,-1))
        logits = torch.stack(logits,axis=0).squeeze(1)
        return logits
    
