

# --- Imports --- #
import torch.utils.data as data
from PIL import Image,ImageFile
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import imghdr
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self,dataset_name, crop_size, train_data_dir):
        super().__init__()
        train_list = train_data_dir + 'trainlist.txt'
        with open(train_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i.split('_')[0] for i in haze_names] #haze_names#
             
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.dataset_name = dataset_name
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        # filename1=self.train_data_dir + 'haze/' + haze_name

        #f = open('check_error.txt','w+')
        #check = imghdr.what(filename1)
        #if check != None:
            #print(filename1)
            #f.write(filename1)
 
            #f.write('\n')
 
            #error_images.append(filename1)
 
        if self.dataset_name=='NH' or self.dataset_name=='dense':
            haze = Image.open(self.train_data_dir + 'haze/' + haze_name) 
            clear = Image.open(self.train_data_dir + 'clear_images/' + gt_name + '_GT.png')
            width, height = haze.size
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size_w,self.size_h))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        elif self.dataset_name=='indoor':
            haze = Image.open(self.train_data_dir + 'haze/' + haze_name) 
            haze=haze.resize((self.size_w, self.size_h))
            clear = Image.open(self.train_data_dir + 'clear_images/' + gt_name + '.png')
            clear=clear.resize((self.size_w, self.size_h)) 
            width, height = haze.size
        elif self.dataset_name=='outdoor':
            haze = Image.open(self.train_data_dir + 'haze/' + haze_name) 
            haze=haze.resize((self.size_w, self.size_h))
            clear = Image.open(self.train_data_dir + 'clear_images/' + gt_name + '.png')
            clear=clear.resize((self.size_w, self.size_h)) 
            width, height = haze.size
        else:
            print('The dataset is not included in this work.')

        haze,gt=self.augData(haze.convert("RGB") ,clear.convert("RGB") )

        # --- Check the channel is 3 or not --- # 
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3: 
            #print(gt_name)
            raise Exception('Bad image channel: {}'.format(gt_name))
        return haze, gt
    def augData(self,data,target):
        #if self.train:
        if 1: 
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __getitem__(self, index):
        res = self.get_images(index)
        return res 
    def __len__(self):
        return len(self.haze_names)

