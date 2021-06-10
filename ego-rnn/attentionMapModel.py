import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class attentionMap(nn.Module):
    def __init__(self, backbone):
        super(attentionMap, self).__init__()
        self.backbone = backbone
        self.backbone.train(False)
        self.params = list(self.backbone.parameters())
        self.weight_softmax = self.params[-2]
    def forward(self, img_variable, img, size_upsample):
        logit, feature_conv, _ = self.backbone(img_variable)

        bz, nc, h, w = feature_conv.size()
        feature_conv = feature_conv.view(bz, nc, h*w)

        h_x = F.softmax(logit, dim=1).data
        probs, idx = h_x.sort(1, True)
        cam_img = torch.bmm(self.weight_softmax[idx[:, 0]].unsqueeze(1), feature_conv).squeeze(1)
        cam_img = F.softmax(cam_img, 1).data
        cam_img = cam_img.cpu().numpy()
        cam_img = cam_img.reshape(h, w)
        cam_img = cam_img - np.min(cam_img)
        cam_img = cam_img / np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)
        output_cam = cv2.resize(cam_img, size_upsample)
        #PScegliere colore più consono con
        #img = np.uint8(img)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR) #https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a01a06e50cd3689f5e34e26daf3faaa39
        heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET) #https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        result = heatmap * 0.3 + img * 0.5
        return result