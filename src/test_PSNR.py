# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData
from swin_unet import UNet_emb #UNet
from utils import validation_PSNR, generate_filelist
import os
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of dehamer from Guo et al. (2022)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'dense', 'indoor','outdoor','our_test'], default='NH')
parser.add_argument('-t', '--test-image-dir', help='test images path', default='./data/classic_test_image/')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path', default='./ckpts/outdoor/PSNR3518_SSIM09860.pt')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
args = parser.parse_args()


val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
# import pdb;pdb.set_trace()
# --- Set dataset-specific hyper-parameters  --- #
if dataset_name == 'NH':
    val_data_dir = './data/valid_NH/'
    ckpts_dir = './ckpts/NH/PSNR2066_SSIM06844.pt'
elif dataset_name == 'dense': 
    val_data_dir = './data/valid_dense/'
    ckpts_dir = './ckpts/dense/PSNR1662_SSIM05602.pt'
elif dataset_name == 'indoor': 
    val_data_dir = './data/valid_indoor/'
    ckpts_dir = './ckpts/indoor/PSNR3663_ssim09881.pt'
elif dataset_name == 'outdoor': 
    val_data_dir = './data/valid_outdoor/'
    ckpts_dir = './ckpts/outdoor/PSNR3518_SSIM09860.pt'
else:
    val_data_dir = args.test_image_dir
    ckpts_dir =  args.ckpts_dir

# prepare .txt file
if not os.path.exists(os.path.join(val_data_dir, 'val_list.txt')):
    generate_filelist(val_data_dir, valid=True)

# --- Gpu device --- #
device_ids =  [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(dataset_name,val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
net = UNet_emb()

  
# --- Multi-GPU --- # 
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(ckpts_dir), strict=False)


# --- Use the evaluation model in testing --- #
net.eval() 
print('--- Testing starts! ---') 
start_time = time.time()
val_psnr, val_ssim = validation_PSNR(net, val_data_loader, device, dataset_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))