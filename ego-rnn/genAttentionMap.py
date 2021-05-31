import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision import transforms
from PIL import Image
import resnetMod
from MyConvLSTMCell import *
import glob
import os
import imageio
from objectAttentionModelConvLSTM import *
from attentionMapModel import 

num_classes = 61 # Classes in the pre-trained model
mem_size = 512
model = attentionModel(num_classes, mem_size)

model_state_dict = 'model_ss_task1.pth' # Weights of the pre-trained model
model.load_state_dict(torch.load(model_state_dict),strict=False)
model_backbone = model.resNet
attentionMapModel = attentionMap(model_backbone).to("cuda")
attentionMapModel.train(False)

for params in attentionMapModel.parameters():
    params.requires_grad = False
    
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess1 = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
])

preprocess2 = transforms.Compose([
    transforms.ToTensor(),
    normalize])
