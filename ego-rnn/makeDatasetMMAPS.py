import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random

root_dir = os.path.join('/content/drive/MyDrive/ML_project/ego-rnn/content/GTEA61', '')
def gen_split(root_dir,train_dataset_folder, stackSize = 5):
    Dataset_RGBFrame = []
    Dataset_MMAPSFrame = []
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
              inst_dir_mmaps = os.path.join(inst_dir, 'mmaps')
              numFrames_rgb = len(glob.glob1(inst_dir_rgb, '*[0-9].png')) # nome dei file per ogni azione [0-9 indica numero generico]
              numFrames_mmaps = len(glob.glob1(inst_dir_mmaps, '*[0-9].png'))
              if numFrames_rgb >= 7 and numFrames_mmaps>=7: # numero di frame sufficiente
                NumFrames.append(numFrames_rgb) # numero di frame x ogni folder
                Labels.append(class_id) # numero della classe del azione
                Dataset_RGBFrame.append(inst_dir_rgb)
                Dataset_MMAPSFrame.append(inst_dir_mmaps)
            class_id += 1
    return Dataset_RGBFrame, Dataset_MMAPSFrame, Labels, NumFrames
class makeDataset(Dataset):
    def __init__(self, root_dir,train_dataset_folder, spatial_transform, seqLen=7, train=True):
        self.dataset_RGBFrame, self.dataset_MMAPSFrame, self.labels, self.numFrames = gen_split(root_dir,train_dataset_folder, 5)
        self.spatial_transform = spatial_transform
        self.train = train
        self.seqLen = seqLen
        self.stackSize = 5
        self.fmt = '.png'
    def __len__(self):
        return len(self.dataset_RGBFrame)
    def __getitem__(self, idx): #dataset[idx]
        vid_nameF = self.dataset_RGBFrame[idx]
        vid_nameM = self.dataset_MMAPSFrame[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        inpSeqM = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False): 
            go_on = False
            j = i
            k=1
            while(go_on==False):
                try:
                  fm_name = vid_nameM + '/' + 'map' + str(int(np.floor(j))).zfill(4) + self.fmt
                  img = Image.open(fm_name)
                  inpSeqM.append(self.spatial_transform(img, inv=False, flow=True))
                  fl_name = vid_nameF + '/' + 'rgb' + str(int(np.floor(j))).zfill(4) + self.fmt
                  img = Image.open(fl_name)
                  inpSeq.append(self.spatial_transform(img.convert('RGB')))
                  go_on = True
                except:
                  if j == numFrame:
                    k=-1
                    j=i-1
                  else:
                    j += k*1
                  continue
        inpSeq = torch.stack(inpSeq, 0)
        inpSeqM = torch.stack(inpSeqM, 0)
        return inpSeq, inpSeqM, label
