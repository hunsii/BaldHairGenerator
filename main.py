import argparse

import torch
import torchvision
import numpy as np
import sys
import os
import dlib
import time
import shutil


from PIL import Image


from models.Embedding import Embedding
from models.AlignmentBaldHair import Alignment

from utils.data_utils import convert_npy_code
toPIL = torchvision.transforms.ToPILImage()


# +
def main(args):    
    ii2s = Embedding(args)
    #
    # ##### Option 1: input folder
    # # ii2s.invert_images_in_W()
    # # ii2s.invert_images_in_FS()

    # ##### Option 2: image path
    # # ii2s.invert_images_in_W('input/face/28.png')
    # # ii2s.invert_images_in_FS('input/face/28.png')
    #
    ##### Option 3: image path list

    # im_path1 = 'input/face/90.png'
    # im_path2 = 'input/face/15.png'
    # im_path3 = 'input/face/117.png'

    im_path1 = os.path.join(args.input_dir, args.im_path1)
    im_set = {im_path1}
    
    if args.W_steps <= 0 and os.path.isfile(args.W_saved_latent):
        print('replace W+ latent code from pSp encoder')
        latent_in = torch.from_numpy(convert_npy_code(np.load(args.W_saved_latent)['latents'])).to(args.device)
        gen_im, _ = ii2s.net.generator([latent_in], input_is_latent=True, return_latents=False)
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()
        
        output_dir = os.path.join(args.output_dir, 'W+')
        os.makedirs(output_dir, exist_ok=True)
        
        im_name = os.path.basename(args.im_path1).split('.')[0]
        latent_path = os.path.join(output_dir, f'{im_name}.npy')
        image_path = os.path.join(output_dir, f'{im_name}.png')
        
        save_im.save(image_path)
        np.save(latent_path, save_latent)
    else:
        ii2s.invert_images_in_W([*im_set])
    ii2s.invert_images_in_FS([*im_set])
    
    align = Alignment(args)
    align.align_images(im_path1, sign=args.sign, align_more_region=False, smooth=args.smooth)
    
    print('done!')


# -
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Barbershop')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/face',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images')
    parser.add_argument('--im_path1', type=str, default='16.png', help='Identity image')
#     parser.add_argument('--im_path2', type=str, default='15.png', help='Structure image')
#     parser.add_argument('--im_path3', type=str, default='117.png', help='Appearance image')
    parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
    parser.add_argument('--verbose', action='store_true', help='Print loss information')
    parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')


    # Embedding loss options
    parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
    parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
    parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
    parser.add_argument('--l_F_lambda', type=float, default=0.1, help='L_F loss multiplier factor')
    parser.add_argument('--seg_lambda', type=float, default=1.0, help='L_F loss multiplier factor')
    parser.add_argument('--W_steps', type=int, default=1100, help='Number of W space optimization steps')
    parser.add_argument('--W_saved_latent', type=str, default='', help='The directory of the images to be inverted')
    parser.add_argument('--use_psp', action='store_true', help='Print loss information')
    
    parser.add_argument('--FS_steps', type=int, default=250, help='Number of W space optimization steps')
    parser.add_argument('--FS_saved_latent', type=str, default='', help='The directory of the images to be inverted')


    # Alignment loss options
    parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
    parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
    parser.add_argument('--align_steps1', type=int, default=350, help='')
    
    args = parser.parse_args()
    main(args)

