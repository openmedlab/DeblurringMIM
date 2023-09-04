from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import os
import torch
from collections import Counter

from torchvision import transforms as T

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            t_size=512,
            need_down=True,
            is_rgb =True,
            need_norm=True,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = [f for f in os.listdir(images_dir) if '.png' in f]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.size = t_size

        # convert str names to class values on masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.need_down = need_down
        if self.need_down:
            self.mask_size = int(self.size / 2)
        else:
            self.mask_size = self.size
        self.is_rgb = is_rgb
        self.need_norm = need_norm

    def __getitem__(self, i):
        # read data
        if self.is_rgb:
            image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print ("image is none,", self.images_fps[i])

        if mask is None:
            print ("mask is none,", self.masks_fps[i])


        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        mask = cv2.resize(mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST )

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        ""
        image = image / 255.0
        mask = mask / 255.0
        
        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1
        
        if self.need_norm:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            image = (image - mean) / std    
            
        if self.is_rgb:
            image = image.transpose(2, 0, 1)
        else:
            image = np.expand_dims(image, 0)
        image = image.astype('float32')

        mask = np.expand_dims(mask, 0).astype('float32')

        return image, mask, self.images_fps[i] [self.images_fps[i].rfind('/') + 1:]

    def __len__(self):
        return len(self.ids)
    

    
    
    
