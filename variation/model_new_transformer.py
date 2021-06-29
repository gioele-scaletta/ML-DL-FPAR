import torch
import resnetMod
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyTransformer2 import *
from ss_task import *

class selfAttentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, num_frames=16):
        super(selfAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.ss_task = ss_task()
        self.weight_softmax = self.resNet.fc.weight
        self.transf = MyTransformer(num_frames)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.conv_f = nn.Conv1d(512, 61, kernel_size=1, stride=1)
        self.fc = nn.Linear(self.mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        nn.init.xavier_normal_(self.conv_f.weight)
        nn.init.constant_(self.conv_f.bias, 0)
        

    def forward(self, inputVariable):
        logits = []
        feats_ss = []
        embeddings = []
        bz, nf, nc_rgb, h, w = inputVariable.size()
        print("input variable: ",inputVariable.size())
        inputVariable = inputVariable.view(-1,nc_rgb,h,w)
        logit, feature_conv, feature_convNBN = self.resNet(inputVariable)
        bz_nf, nc_resnet, h, w = feature_conv.size()    #bz = batch size    #nc = number of channels    #h = height   #w = width
        print("feature conv", feature_conv.size())
        feature_conv1 = feature_conv.view(bz_nf, nc_resnet, h*w)    #make vector from matrix (hxw -> h*w)
        probs, idxs = logit.sort(1, True)   #sort in order from highest to lowest the score of each class in each batch
        class_idx = idxs[:, 0]    
        cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
        attentionMAP = F.softmax(cam.squeeze(1), dim=1)
        attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
        attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
        print("attention feat:",attentionFeat.size())
        embeddings = torch.squeeze(torch.squeeze(self.avgpool(attentionFeat),3),2)
        embeddings = embeddings.view(bz,nf,-1)
        logit = self.transf(embeddings)
        print("logit",logit.size())
        final_logit = self.fc(logit.cuda())
        print("final logit",final_logit.size())
        ss_task_feats = self.ss_task(feature_conv) # a tensor of size [32,7*7] is returned
        print("ss task feats",ss_task_feats.size())
        feats_ss = ss_task_feats.view(bz, nf, 7*7) #now that it is a regression problem no more 2 and no more softmax needed
        print("feats self super", feats_ss.size())
        print("_"*50)
        return final_logit, feats_ss
