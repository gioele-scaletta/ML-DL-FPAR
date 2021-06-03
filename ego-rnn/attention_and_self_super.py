import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from ss_task import *


class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        self.ss_task = ss_task()
    
    #inputVariable.Size([7, 32, 3, 224, 224]) 
    #feature_conv.Size([32, 512, 7, 7])
    
    def forward(self, inputVariable, stage):
        feats_ss = []
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),    #inputVariable.size(1) = size of batch
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):    #inputVariable.size(0) = number of frames
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])    #logit = prediction of the 1000 classes[1000 x 32]   #feature_conv = weights of conv4   #feature_convNBN = weights of conv4 w/out BN
            bz, nc, h, w = feature_conv.size()    #bz = batch size    #nc = number of channels    #h = height   #w = width
            feature_conv1 = feature_conv.view(bz, nc, h*w)    #make vector from matrix (hxw -> h*w)
            probs, idxs = logit.sort(1, True)   #sort in order from highest to lowest the score of each class in each batch
            class_idx = idxs[:, 0]    
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state = self.lstm_cell(attentionFeat, state)
            
            if stage == 2:
                ss_task_feats = self.ss_task(feature_conv)
                ss_task_feats = ss_task_feats.view(feature_conv.size(0), 7*7, 2)
                ss_task_feats = nn.functional.softmax(ss_task_feats,2)
                feats_ss.append(ss_task_feats)
        
        if stage == 2:
            feats_ss = torch.stack(feats_ss,0)
            
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, feats_ss

