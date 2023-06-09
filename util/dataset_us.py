from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
import random
import torch
import time

class CorruptDataset(BaseDataset):
    def __init__(
            self,
            images_dir_corrupt,
            images_dir_original,
            input_size, is_train ):
        self.is_train = is_train
        
        self.dirs = os.listdir(images_dir_corrupt + "/train/")
        print ("self.dirs:", self.dirs)
        
        self.ids = []
        if self.is_train:
            for dir in self.dirs:
                self.ids += [ "train/" + dir + "/" + name for name in
                         os.listdir(images_dir_corrupt + "/train/" + dir)]
        else:
            for dir in self.dirs:
                self.ids += [ "val/" + dir + "/" + name for name in
                         os.listdir(images_dir_corrupt + "/val/" + dir)]
        
        print ("len(self.ids):", len(self.ids))
   
        self.images_corr = [os.path.join(images_dir_corrupt, image_id) for image_id in self.ids]
        self.images_orig = [os.path.join(images_dir_original, image_id) for image_id in self.ids]

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_val = transforms.Compose([
            # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, i):
        # img_corr = cv2.imread(self.images_corr[i], cv2.IMREAD_COLOR) / 255.0
        # img_orig = cv2.imread(self.images_orig[i], cv2.IMREAD_COLOR) /255.0

        # img_corr = img_corr.transpose(2, 0, 1)
        # img_orig = img_orig.transpose(2, 0, 1)
        img_corr = Image.open(self.images_corr[i]).convert('RGB')
        img_orig = Image.open(self.images_orig[i]).convert('RGB')
        seed = time.time()
        if self.is_train:
            torch.manual_seed(seed) 
            img_corr = self.transform_train(img_corr)
            torch.manual_seed(seed) 
            img_orig = self.transform_train(img_orig)
        else:
            torch.manual_seed(seed) 
            img_corr = self.transform_val(img_corr)
            torch.manual_seed(seed) 
            img_orig = self.transform_val(img_orig)
        
        return img_corr, img_orig
    
    
    def __len__(self):
        return len(self.ids)