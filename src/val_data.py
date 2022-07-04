"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch
import os

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, dataset_name,val_data_dir):
        super().__init__() 
        self.dataset_name = dataset_name
        val_list = os.path.join(val_data_dir, 'val_list.txt')
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            if self.dataset_name=='NH' or self.dataset_name=='dense':
                gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names] #haze_names#
            elif self.dataset_name=='indoor' or self.dataset_name=='outdoor':
                gt_names = [i.split('_')[0] + '.png' for i in haze_names]   
            else:
                gt_names = None 
                print('The dataset is not included in this work.')  
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.data_list=val_list
    def get_images(self, index):
        haze_name = self.haze_names[index]

        # build the folder of validation/test data in our way
        if os.path.exists(os.path.join(self.val_data_dir, 'input')):
            haze_img = Image.open(os.path.join(self.val_data_dir, 'input', haze_name))
            if os.path.exists(os.path.join(self.val_data_dir, 'gt')) :
                gt_name = self.gt_names[index]
                gt_img = Image.open(os.path.join(self.val_data_dir, 'gt', gt_name)) ##   
                a = haze_img.size
                a_0 =a[1] - np.mod(a[1],16)
                a_1 =a[0] - np.mod(a[0],16)            
                haze_crop_img = haze_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = gt_img.crop((0, 0, 0 + a_1, 0+a_0))
                transform_haze = Compose([ToTensor() , Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
                transform_gt = Compose([ToTensor()])
                haze_img = transform_haze(haze_crop_img)
                gt_img = transform_gt(gt_crop_img)
            else: 
                # the inputs is used to calculate PSNR.
                a = haze_img.size
                a_0 =a[1] - np.mod(a[1],16)
                a_1 =a[0] - np.mod(a[0],16)            
                haze_crop_img = haze_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = haze_crop_img
                transform_haze = Compose([ToTensor() , Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
                transform_gt = Compose([ToTensor()])
                haze_img = transform_haze(haze_crop_img)
                gt_img = transform_gt(gt_crop_img) 
        # Any folder containing validation/test images
        else:
            haze_img = Image.open(os.path.join(self.val_data_dir, haze_name))
            a = haze_img.size
            a_0 =a[1] - np.mod(a[1],16)
            a_1 =a[0] - np.mod(a[0],16)            
            haze_crop_img = haze_img.crop((0, 0, 0 + a_1, 0+a_0))
            gt_crop_img = haze_crop_img
            transform_haze = Compose([ToTensor() , Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
            transform_gt = Compose([ToTensor()])
            haze_img = transform_haze(haze_crop_img)
            gt_img = transform_gt(gt_crop_img)           
        return haze_img, gt_img, haze_name


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)

