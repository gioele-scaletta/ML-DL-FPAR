import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random

# prima di tutto crea la classe con makeDataset, si fa dare la directory con i training frames


def gen_split(root_dir, stackSize):
    Dataset = [] #Path to RGB frame
    Labels = [] #Labels
    NumFrames = [] #
    root_dir = os.path.join(root_dir, 'processed_frames2')
    for dir_user in sorted(os.listdir(root_dir)):
        print('dir_user' + dir_user)
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        print('dir' + str(dir))
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            
            insts = sorted(os.listdir(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst)
                    inst_dir = os.path.join(inst_dir, 'rgb')
                    numFrames = len(glob.glob1(inst_dir, '*.png'))
                    if numFrames >= stackSize:
                        Dataset.append(inst_dir)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):

        self.images, self.labels, self.numFrames = gen_split(root_dir, 5)
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
