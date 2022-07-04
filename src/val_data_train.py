# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch


# --- Validation/test dataset --- #
class ValData_train(data.Dataset):
    def __init__(self,dataset_name,crop_size, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'val_list.txt' 
        self.dataset_name = dataset_name      
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            if self.dataset_name=='NH' or self.dataset_name=='dense':
                gt_names = [i.split('_')[0] + '_GT.png' for i in haze_names] #haze_names#
            elif self.dataset_name=='indoor' or self.dataset_name=='outdoor':
                gt_names = [i.split('_')[0] + '.png' for i in haze_names]   
            else:
                print('The dataset is not included in this work.')            
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.data_list=val_list  
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
    def get_images(self, index):
        haze_name = self.haze_names[index] 
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'input/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name)
          
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Resize((self.size_w, self.size_h)), Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))])
        transform_gt = Compose([ToTensor(),Resize((self.size_w, self.size_h))])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img) 
         
        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res
    def __len__(self): 
        return len(self.haze_names)

 