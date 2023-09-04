import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.nn as nn



model = smp.Unet( encoder_name="convmae", encoder_weights='/qingbo/ConvMAE-main/pretrain_zk/0425_dconvmae_base_gaussian/checkpoint-799.pth',  
                 in_channels=3, classes=1, activation='sigmoid')

print (model)

model = model.to('cuda')

x = torch.randn([1, 3, 512, 512], device='cuda')


y = model(x)

print (y.shape)

