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
from attentionMapModel import *

num_classes = 61 # Classes in the pre-trained model
mem_size = 512
model = attentionModel(num_classes, mem_size)

model_state_dict = '/content/drive/MyDrive/ML_project/Risultati_e_modelli/SS_Task/model_ss_task1.pth' # Weights of the pre-trained model
model.load_state_dict(torch.load(model_state_dict),strict=False)
model_backbone = model.resNet
attentionMapModel = attentionMap(model_backbone).to("cuda")
attentionMapModel.train(False)

for params in attentionMapModel.parameters():
    params.requires_grad = False

def applyAttentionToRGB(rgb_folder, output_folder):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
  preprocess1 = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
  preprocess2 = transforms.Compose([transforms.ToTensor(),normalize])

  img_array = [rgb_folder+x for x in os.listdir(rgb_folder) if x.endswith("png")]
  img_array.sort() # tiene ordine se frame numerati
  n_rgb_frames=0

  img_array_res_np = []
  for img in img_array:
    fl_name_in = img
    fl_name_out = output_folder+ img.split("/")[-1]

    img_pil = Image.open(fl_name_in)
    img_pil1 = preprocess1(img_pil)
    img_size = img_pil1.size
    size_upsample = (img_size[0], img_size[1])
    img_tensor = preprocess2(img_pil1)
    img_variable = Variable(img_tensor.unsqueeze(0).cuda())
    img = np.asarray(img_pil1)
    attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
    if cv2.imwrite(fl_name_out, attentionMap_image):
      n_rgb_frames=n_rgb_frames+1
      img_array_res_np.append(np.uint8(attentionMap_image))
  return n_rgb_frames,img_array_res_np

class makeAttentionImage():
    def __init__(self, rgb_folder, output_folder):
        self.folder_input=rgb_folder
        self.output_folder=output_folder
        self.nOfFrames,self.imageWithAttentionNP = applyAttentionToRGB(rgb_folder,output_folder)


inputPath = '/content/video/' #Folder that cointains the RGB Frames 
outputPath = inputPath + 'res/'
video1 = makeAttentionImage(inputPath,outputPath)
imageio.mimsave(outputPath+'file.gif',video1.imageWithAttentionNP,fps=8)
