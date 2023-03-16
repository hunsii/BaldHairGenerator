import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset
from losses.embedding_loss import EmbeddingLossBuilder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import PIL
import torchvision
from utils.data_utils import convert_npy_code
import time
import joblib
import cv2

toPIL = torchvision.transforms.ToPILImage()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 2
color = (0, 0, 255)
text_x = 20
text_y = 100


# +
class Embedding(nn.Module):

    def __init__(self, opts):
        super(Embedding, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.setup_embedding_loss_builder()

    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)

    def setup_W_optimizer(self, saved_latent_path = None):

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        latent = []
        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            if os.path.isfile(saved_latent_path):
                print('-'*80)
                print('Loading W latent...')
                
                latent_W = torch.from_numpy(convert_npy_code(np.load(saved_latent_path)['latents'])).to(self.opts.device)
                latent_W = latent_W[0]
                for i in range(latent_W.shape[0]):
                    tmp = latent_W[i]
                    tmp.requires_grad = True
                    latent.append(tmp)
                print('Loading W latent is success')
            else:
                print('Inisializing W latent')
                for i in range(self.net.layer_num):
                    tmp = self.net.latent_avg.clone().detach().cuda()
                    tmp.requires_grad = True
                    latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)
            return optimizer_W, latent



    def setup_FS_optimizer(self, latent_W, F_init, ref_name = None):

        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
#         if ref_name:
#             print('-'*80)
#             print('Loading FS latent...')
#             latent_FS_path = os.path.join(self.opts.output_dir, 'FS', f'{ref_name[0]}_default.npz')
#         else:
#             latent_FS_path = None
        
#         if os.path.isfile(latent_FS_path):
#             latent_FS = np.load(latent_FS_path)
#             latent_F = torch.from_numpy(convert_npy_code(latent_FS['latent_F'])).to(self.opts.device)
#             latent_W = torch.from_numpy(convert_npy_code(latent_FS['latent_in'])).to(self.opts.device)
#             print('Loading FS latent is success!!')
        for i in range(self.net.layer_num):

            tmp = latent_W[0, i].clone()

            if i < self.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True

            latent_S.append(tmp)

        optimizer_FS = opt_dict[self.opts.opt_name](latent_S[self.net.S_index:] + [latent_F], lr=self.opts.learning_rate)
        return optimizer_FS, latent_F, latent_S




    def setup_dataloader(self, image_path=None):

        self.dataset = ImagesDataset(opts=self.opts,image_path=image_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print("Number of images: {}".format(len(self.dataset)))

    def setup_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder(self.opts)


    def invert_images_in_W(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        device = self.opts.device
        
        ibar = tqdm(self.dataloader, desc='Images')
        
        for ref_im_H, ref_im_L, ref_name in ibar:
            # code for video output
            os.makedirs('output/loss/W+/', exist_ok=True)
            # Create a VideoWriter object to write the video to a file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_rate = 60
            video_writer = cv2.VideoWriter(f'output/loss/W+/{ref_name[0]}.mp4', fourcc, frame_rate, (1024, 1024))

            log_loss = open(f'output/loss/W+/{ref_name[0]}.txt', 'w')
            log_loss.write("l2,percep,p-norm,item\n")
            
            optimizer_W, latent = self.setup_W_optimizer(self.opts.W_saved_latent)
            pbar = tqdm(range(self.opts.W_steps), desc='Embedding', leave=False)
            for step in pbar:
                optimizer_W.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)
                gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
                tensor_image = ((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)
                
                # convert the tensor to a PIL image
                pil_image = toPIL(tensor_image)
                # convert the PIL image to an OpenCV image
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                # Add text to the image
                text = f"{step}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.putText(opencv_image, text, (text_x, text_y), font, font_scale, color, thickness)

                
                video_writer.write(opencv_image)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }
                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                # log loss
                log_loss.write(f'{loss_dic["l2"]},{loss_dic["percep"]},{loss_dic["p-norm"]},{loss.item()}\n')
                loss.backward()
                optimizer_W.step()
                
                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))

                if self.opts.save_intermediate and step % self.opts.save_interval== 0:
                    self.save_W_intermediate_results(ref_name, gen_im, latent_in, step)
            # Release the video writer
            log_loss.close()
            video_writer.release()
            self.save_W_results(ref_name, gen_im, latent_in)

    def invert_images_in_FS(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        output_dir = self.opts.output_dir
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')
        for ref_im_H, ref_im_L, ref_name in ibar:

            latent_W_path = os.path.join(output_dir, 'W+', f'{ref_name[0]}.npy')
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(device)
            F_init, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

            # Create a VideoWriter object to write the video to a file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_rate = 60
            video_writer = cv2.VideoWriter(f'output/loss/FS/{ref_name[0]}.mp4', fourcc, frame_rate, (1024, 1024))
            
            os.makedirs('output/loss/FS/', exist_ok=True)
            log_loss = open(f'output/loss/FS/{ref_name[0]}.txt', 'w')
            log_loss.write("l2,percep,p-norm,item\n")
            
            optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init, ref_name)
            pbar = tqdm(range(self.opts.FS_steps), desc='Embedding', leave=False)
            for step in pbar:
                optimizer_FS.zero_grad()
                latent_in = torch.stack(latent_S).unsqueeze(0)
                gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                               start_layer=4, end_layer=8, layer_in=latent_F)
                tensor_image = ((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)
                # convert the tensor to a PIL image
                pil_image = toPIL(tensor_image)
                # convert the PIL image to an OpenCV image
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                # Add text to the image
                text = f"{step}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.putText(opencv_image, text, (text_x, text_y), font, font_scale, color, thickness)                
                video_writer.write(opencv_image)
                
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }
                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                log_loss.write(f'{loss_dic["l2"]},{loss_dic["percep"]},{loss_dic["p-norm"]},{loss.item()}\n')

                loss.backward()
                optimizer_FS.step()


                if self.opts.verbose:
                    pbar.set_description(
                        'Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}, L_F loss: {:.3f}'
                        .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm'], loss_dic['l_F']))
            log_loss.close()
            video_writer.release()
            self.save_FS_results(ref_name, gen_im, latent_in, latent_F)


    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.net.cal_l_F(latent_F, F_init)
            loss_dic['l_F'] = l_F
            loss += l_F

        return loss, loss_dic



    def save_W_results(self, ref_name, gen_im, latent_in):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        output_dir = os.path.join(self.opts.output_dir, 'W+')
        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npy')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)



    def save_W_intermediate_results(self, ref_name, gen_im, latent_in, step):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        intermediate_folder = os.path.join(self.opts.output_dir, 'W+', ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)


    def save_FS_results(self, ref_name, gen_im, latent_in, latent_F):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        output_dir = os.path.join(self.opts.output_dir, 'FS')
        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npz')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(),
                 latent_F=latent_F.detach().cpu().numpy())


    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
