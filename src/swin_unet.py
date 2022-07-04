import torch
import torch.nn as nn
from swin import SwinTransformer
import torch.nn.functional as F

from MIRNet import *
class UNet_emb(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3,bias=False):
        """Initializes U-Net."""

        super(UNet_emb, self).__init__()
        self.embedding_dim = 3
        #self.conv0 = nn.Conv2d(3, self.embedding_dim, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(256*3, 256, 3, stride=1, padding=1) 
        self.conv1_1 = nn.Conv2d(384, 256, 3, stride=1, padding=1)         
        self.conv1_2 = nn.Conv2d(384, 256, 3, stride=1, padding=1)    
        self.conv2 = nn.Conv2d(128*3, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv2_2= nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64*3, 64, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(96, 64, 3, stride=1, padding=1) 
        self.conv3_2 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(48, 24, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(96, 48, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(48, 24, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(24, 12, 3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(12, 12, 3, stride=1, padding=1)
        self.in_chans = 3
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ReLU=nn.ReLU(inplace=True)
        self.IN_1=nn.InstanceNorm2d(48, affine=False)
        self.IN_2=nn.InstanceNorm2d(96, affine=False)
        self.IN_3=nn.InstanceNorm2d(192, affine=False)
        self.PPM1 = PPM(32, 8, bins=(1,2,3,4))
        self.PPM2 = PPM(64, 16, bins=(1,2,3,4))
        self.PPM3 = PPM(128, 32, bins=(1,2,3,4))
        self.PPM4 = PPM(256, 64, bins=(1,2,3,4))
        
        self.MSRB1=MSRB(256, 3, 1, 2,bias)
        self.MSRB2=MSRB(128, 3, 1, 2,bias)
        self.MSRB3=MSRB(64, 3, 1, 2,bias)
        self.MSRB4=MSRB(32, 3, 1, 2,bias)

        # 27,565,242
        self.swin_1 = SwinTransformer(pretrain_img_size=224,
                                    patch_size=2,
                                    in_chans=3,
                                    embed_dim=96,
                                    depths=[2, 2, 2],
                                    num_heads=[3, 6, 12], 
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True, 
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0., 
                                    drop_path_rate=0.2,
                                    norm_layer=nn.LayerNorm, 
                                    ape=False,
                                    patch_norm=True,
                                    out_indices=(0, 1, 2),
                                    frozen_stages=-1,
                                    use_checkpoint=False)

        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        
        self.E_block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        
        self.E_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))


            
        self._block1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
        
        self._block5= nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))
            
        self._block6= nn.Sequential(
            nn.Conv2d(46, 23, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(23, 23, 3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2))
            
        self._block7= nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))     
                   
        # Initialize weights
        #self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        swin_in = x #96,192,384,768
        swin_out_1=self.swin_1(swin_in)      
        #swin_out = self.swin(swin_in)
        # Encoder 
        swin_input_1=self.E_block1(swin_in)#32    
        swin_input_1=self.PPM1(swin_input_1)  

        swin_input_2=self.E_block2(swin_input_1)#64 
        swin_input_2=self.PPM2(swin_input_2)     

        swin_input_3=self.E_block3(swin_input_2)#128
        swin_input_3=self.PPM3(swin_input_3) 


        swin_input_4=self.E_block4(swin_input_3)#256
        swin_input_4=self.PPM4(swin_input_4) 
        #swin_input_5=self.E_block5(swin_input_4)#512
        #import pdb 
        #pdb.set_trace() 

        # transformer 
        upsample1 = self._block1(swin_input_4)#256
        #upsample1=[self.MSRB1(upsample1)]
        #import pdb 
        #pdb.set_trace() 
        beta_1 = self.conv1_1(swin_out_1[2])
        gamma_1 = self.conv1_2(swin_out_1[2])
        swin_input_3_refine=self.IN_3(swin_input_3)*beta_1+gamma_1#128
        concat3 = torch.cat((swin_input_3,swin_input_3_refine,upsample1), dim=1)#256+256+256==512
        decoder_3 = self.ReLU(self.conv1(concat3)) #256
        upsample3 = self._block3(decoder_3)#128
        upsample3=self.MSRB2(upsample3)
        
        beta_2 = self.conv2_1(swin_out_1[1])
        gamma_2 = self.conv2_2(swin_out_1[1])
        swin_input_2_refine=self.IN_2(swin_input_2)*beta_2+gamma_2 #64
        concat2 = torch.cat((swin_input_2,swin_input_2_refine,upsample3), dim=1)#128+128+128=256
        decoder_2 = self.ReLU(self.conv2(concat2))#128
        upsample4 = self._block4(decoder_2)#64
        upsample4=self.MSRB3(upsample4)

        beta_3 = self.conv3_1(swin_out_1[0])
        gamma_3 =self.conv3_2(swin_out_1[0]) 
        swin_input_1_refine=self.IN_1(swin_input_1)*beta_3+gamma_3 #32
        concat1 = torch.cat((swin_input_1,swin_input_1_refine,upsample4), dim=1)#64+64+64=128
        decoder_1 = self.ReLU(self.conv3(concat1))#64
        upsample5 = self._block5(decoder_1)#32
        #upsample5=self.MSRB4(upsample5)
        #decoder_0_1 = self.ReLU(self.conv4_1(swin_out_1[0]))#48 
        #decoder_0_2 = self.ReLU(self.conv4_2(decoder_0_1))#48
        #decoder_0_3 = self.ReLU(self.conv4_3(decoder_0_2))#24
        #decoder_0_4 = self.ReLU(self.conv4_4(decoder_0_3))#12
        #concat0 = torch.cat((upsample5,decoder_0_1), dim=1)#48+48=96
        decoder_0 = self.ReLU(self.conv4(upsample5))#48
        #upsample6 = self._block6(decoder_0)#48

        #Refine_1=self._block6(decoder_0)
        result=self._block7(decoder_0)#23
        # result=self._block7(self._block6(Refine_2))   
        # concat2 = torch.cat((upsample2, swin_out), dim=1)
        #concat2 = upsample2
        #upsample0 = self._block5(upsample1)
        #concat1 = torch.cat((upsample1, x, swin_in), dim=1)

        # Final activation
        return result
           
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                #nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)


    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))   #what is f(x)
        return torch.cat(out, 1)         
          
class UNet(nn.Module): 
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3): 
        """Initializes U-Net."""

        super(UNet, self).__init__()
        self.embedding_dim = 3
        self.conv0 = nn.Conv2d(3, self.embedding_dim, 3, stride=1, padding=1) 
        self.conv1 = nn.Conv2d(768, 768, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.in_chans = 3
        # 27,565,242
        self.swin = SwinTransformer(pretrain_img_size=224,
                                    patch_size=4,
                                    in_chans=self.in_chans,
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.2,
                                    norm_layer=nn.LayerNorm, 
                                    ape=False,
                                    patch_norm=True,
                                    out_indices=(0, 1, 2, 3),
                                    frozen_stages=-1,
                                    use_checkpoint=False)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self._block1 = nn.Sequential(
            nn.Conv2d(768, 768, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(768, 384, 3, stride=2, padding=1, output_padding=1))

        self._block2 = nn.Sequential(
            nn.Conv2d(768, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 3, stride=2, padding=1, output_padding=1))

        self._block3 = nn.Sequential(
            nn.Conv2d(384, 192, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 3, stride=2, padding=1, output_padding=1))
        
        self._block4 = nn.Sequential(
            nn.Conv2d(192, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        
        self._block5 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels + self.embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()
        print('the number of swin parameters: {}'.format(
            sum([p.data.nelement() for p in self.swin.parameters()])))


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # swin_in = self.conv0(x)
        swin_in = x
        swin_out = self.swin(swin_in)

        # Decoder
        swin_out_3 = self.conv1([3])
        upsample5 = self._block1(swin_out_3)

        swin_out_2 = self.conv2(swin_out[2])
        concat5 = torch.cat((upsample5, swin_out_2), dim=1)
        upsample4 = self._block2(concat5)

        swin_out_1 = self.conv3(swin_out[1])
        concat4 = torch.cat((upsample4, swin_out_1), dim=1)
        upsample3 = self._block3(concat4)

        swin_out_0 = self.conv4(swin_out[0])
        concat3 = torch.cat((upsample3, swin_out_0), dim=1)
        upsample2 = self._block4(concat3)

        # concat2 = torch.cat((upsample2, swin_out), dim=1)
        concat2 = upsample2
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x, swin_in), dim=1)

        # Final activation
        return self._block6(concat1)
