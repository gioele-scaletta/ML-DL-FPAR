import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random

def gen_split(root_dir,train_dataset_folder, stackSize = 5):
    Dataset_RGBFrame = []
    Labels = [] #Labels
    NumFrames = [] #
    root_dir = os.path.join(root_dir, 'processed_frames2')
    for dir_user in train_dataset_folder:
        print('Splittin in ' + dir_user + ' folder')
        class_id = 0
        dir = os.path.join(root_dir, dir_user) #Folder effettiva dove mettere i file
        action_sorted = sorted(os.listdir(dir))
        for target in action_sorted:
          if target.startswith(".DS"):
            continue
          dir1 = os.path.join(dir, target) #/ego-rnn/content/GTEA61/flow_x_processed/S4/close_peanut ,/ego-rnn/content/GTEA61/flow_x_processed/S4/close_mustard ..
          insts = sorted(os.listdir(dir1)) # folder 1, 2 ,3 in each action
          if insts != []:
            for inst in insts:
              if inst.startswith(".DS"):
                continue
              inst_dir = os.path.join(dir1, inst)
              inst_dir_rgb = os.path.join(inst_dir, 'rgb')
              numFrames_rgb = len(glob.glob1(inst_dir_rgb, '*[0-9].png')) # nome dei file per ogni azione [0-9 indica numero generico]
              if numFrames_rgb >= 16: # numero di frame sufficiente
                NumFrames.append(numFrames_rgb) # numero di frame x ogni folder
                Labels.append(class_id) # numero della classe del azione
                Dataset_RGBFrame.append(inst_dir_rgb)
            class_id += 1
    return Dataset_RGBFrame, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir,train_dataset_folder, spatial_transform, seqLen=7, train=True):
        self.dataset_RGBFrame, self.labels, self.numFrames = gen_split(root_dir,train_dataset_folder, 7)
        self.spatial_transform = spatial_transform
        self.train = train
        self.seqLen = seqLen
        self.stackSize = 7
        self.fmt = '.png'

    def __len__(self):
        return len(self.dataset_OpticalFlowX)

    def __getitem__(self, idx): #dataset[idx]
        vid_nameF = self.dataset_RGBFrame[idx]
        
        label = self.labels[idx]
        numFrame = self.numFrames[idx]

        inpSeq = []

        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False): 
            fl_name = vid_nameF + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
