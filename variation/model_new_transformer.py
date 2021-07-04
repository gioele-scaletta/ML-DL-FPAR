import torch
import resnetMod
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyTransformer import *
from ss_task import *

class selfAttentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=610, num_frames=16):
        super(selfAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.ss_task = ss_task()
        self.weight_softmax = self.resNet.fc.weight
        self.transf = MyTransformer(num_frames)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.conv_f = nn.Conv1d(610, 61, kernel_size=1, stride=1)
        self.fc = nn.Linear(self.mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        nn.init.xavier_normal_(self.conv_f.weight)
        nn.init.constant_(self.conv_f.bias, 0)
        

    def forward(self, inputVariable):
        logits = []
        feats_ss = []
        embeddings = []
        bz, nf, nc_rgb, h, w = inputVariable.size()
        #print(inputVariable.size())
        inputVariable = inputVariable.view(-1,nc_rgb,h,w)
        logit, feature_conv, feature_convNBN = self.resNet(inputVariable)
        bz_nf, nc_resnet, h, w = feature_conv.size()    #bz = batch size    #nc = number of channels    #h = height   #w = width
        feature_conv1 = feature_conv.view(bz_nf, nc_resnet, h*w)    #make vector from matrix (hxw -> h*w)
        probs, idxs = logit.sort(1, True)   #sort in order from highest to lowest the score of each class in each batch
        class_idx = idxs[:, 0]    
        cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
        attentionMAP = F.softmax(cam.squeeze(1), dim=1)
        #print(attentionMAP.size())
        attentionMAP = attentionMAP.view(bz,nf, -1)
        #attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
        embeddings = torch.squeeze(torch.squeeze(self.avgpool(feature_convNBN),3),2) # also have to try max pooling
        embeddings = embeddings.view(bz,nf,-1)
        ss_task_feats = self.ss_task(feature_conv) # a tensor of size [32,7*7] is returned
        feats_ss = ss_task_feats.view(bz, nf, 7*7) #now that it is a regression problem no more 2 and no more softmax needed
        emb_CAM = torch.cat((embeddings, attentionMAP), -1)
        embeddings_CAM_ss = torch.cat((emb_CAM, feats_ss), -1)
        #print(embeddings_CAM_ss.size())
        logit = self.transf(embeddings_CAM_ss)
        final_logit = self.fc(logit.cuda())
        return final_logit, feats_ss
